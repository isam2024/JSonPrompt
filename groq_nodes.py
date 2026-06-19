"""Groq-backed scene-composition JSON generator for ComfyUI-JSONPrompt.

Same Ideogram4 scene-composition schema and system prompt as the local-GGUF node, but the
inference runs on Groq's cloud. With openai/gpt-oss-120b (or -20b) and strict structured
outputs, Groq constrains decoding at the token level to the JSON schema — the cloud
equivalent of the local llama.cpp grammar — so the key set / types / enums are guaranteed.

Why a separate node: keeps the local node's widget order untouched (ComfyUI saves widget
values by position), and the two backends have genuinely different knobs.

No third-party dependency: posts to Groq's OpenAI-compatible endpoint with stdlib urllib,
so nothing needs installing into the ComfyUI Python. (The `groq` SDK would also work.)
"""

import json
import os
import re
import urllib.request
import urllib.error

# Reuse the schema, system prompt, and flattener from the local node — single source of truth.
from .nodes import SCHEMA_PRESETS, DEFAULT_SYSTEM_PROMPT, _flatten_to_prompt

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Models exposed in the dropdown. Only the gpt-oss pair supports strict schema (constrained
# decoding); the rest fall back to json_object mode (valid JSON, no schema enforcement).
GROQ_MODELS = [
    "openai/gpt-oss-120b",   # strict schema supported — default
    "openai/gpt-oss-20b",    # strict schema supported
    "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
]
_STRICT_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}

# JSON Schema keywords Groq's strict mode does not document/support. Stripped before sending
# so the request is always accepted. Structure/keys/enums are still enforced; string-length
# and array-length discipline is delegated to the system prompt + post-pass below.
_UNSUPPORTED_SCHEMA_KEYS = {
    "maxLength", "minLength", "pattern", "format",
    "minItems", "maxItems", "minimum", "maximum", "multipleOf",
}


# A fixed-length array (the 4-int bbox) cannot be enforced in strict mode once minItems/
# maxItems are stripped — verified live: gpt-oss-120b then emits 1-element bboxes. Strict mode
# DOES guarantee `required` object fields, so we send bbox as an object with 4 required ints
# and convert it back to the canonical [x_min, y_min, x_max, y_max] array afterward.
_BBOX_KEYS = ["x_min", "y_min", "x_max", "y_max"]
# Output order: x-first [x_min, y_min, x_max, y_max]. Empirically verified (2026-06-19) that the
# user's KJNodes Prompt Builder import_json pipeline consumes x-first — a y-first array rendered
# every element rotated 90° (upright subjects came out lying horizontal). Keep x-first.
_BBOX_OUTPUT_ORDER = ["x_min", "y_min", "x_max", "y_max"]


def _is_fixed_int_quad(s):
    return (
        isinstance(s, dict)
        and s.get("type") == "array"
        and isinstance(s.get("items"), dict)
        and s["items"].get("type") == "integer"
        and s.get("minItems") == 4
        and s.get("maxItems") == 4
    )


# Appended to the system prompt whenever the bbox rewrite fires, so the model's generation
# matches the rewritten schema (Groq validates the full schema post-hoc and 400s on mismatch).
_BBOX_OBJECT_OVERRIDE = """

# bbox format OVERRIDE (read carefully — supersedes any array form above)

Emit each element's `bbox` as a JSON OBJECT with four integer fields, NOT an array:

  "bbox": { "x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int> }

on a 1000×1000 canvas, origin top-left, x rightward, y downward. Satisfy
0 ≤ x_min < x_max ≤ 1000 and 0 ≤ y_min < y_max ≤ 1000."""


def _groqify_schema(schema, state):
    """Recursively drop keywords Groq strict mode doesn't support, and rewrite the fixed
    4-int bbox array as a 4-required-field object (the only way to guarantee its arity).
    Sets state['bbox_rewritten'] when a bbox conversion fires."""
    if isinstance(schema, dict):
        if _is_fixed_int_quad(schema):
            state["bbox_rewritten"] = True
            return {
                "type": "object",
                "additionalProperties": False,
                "properties": {k: {"type": "integer"} for k in _BBOX_KEYS},
                "required": list(_BBOX_KEYS),
            }
        return {
            k: _groqify_schema(v, state)
            for k, v in schema.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_groqify_schema(v, state) for v in schema]
    return schema


def _bbox_objects_to_arrays(data):
    """Inverse of the bbox rewrite: turn any {x_min,y_min,x_max,y_max} object back into the
    canonical 4-int array, matching the local node's output exactly."""
    if isinstance(data, dict):
        if set(data.keys()) == set(_BBOX_KEYS) and all(
            isinstance(data.get(k), int) for k in _BBOX_KEYS
        ):
            return [data[k] for k in _BBOX_OUTPUT_ORDER]
        return {k: _bbox_objects_to_arrays(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_bbox_objects_to_arrays(v) for v in data]
    return data


_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _normalize_hex_in_place(data):
    """Uppercase + '#'-prefix any 6-hex string the model emitted, since the strict request no
    longer carries the hex `pattern`. Touches strings only; leaves everything else alone."""
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = _normalize_hex_in_place(v)
        return data
    if isinstance(data, list):
        return [_normalize_hex_in_place(v) for v in data]
    if isinstance(data, str):
        s = data.strip()
        if _HEX_RE.match(s):
            return "#" + s.lstrip("#").upper()
        return data
    return data


class GroqJSONPromptGenerator:
    """Generate the Ideogram4 scene-composition JSON via Groq (cloud, schema-constrained)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (GROQ_MODELS,),
                "user_prompt": (
                    "STRING",
                    {"multiline": True, "default": "a lone lighthouse keeper during a storm"},
                ),
                "schema_preset": (list(SCHEMA_PRESETS.keys()) + ["custom"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "max_completion_tokens": ("INT", {"default": 4096, "min": 256, "max": 32768}),
            },
            "optional": {
                "reasoning_effort": (["low", "medium", "high"], {"default": "medium"}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT},
                ),
                "custom_schema": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "JSON Schema used when schema_preset = custom"},
                ),
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "Groq API key. Leave blank to use the "
                     "GROQ_API_KEY environment variable (preferred — keeps it out of the "
                     "saved workflow)."},
                ),
                "base_url": (
                    "STRING",
                    {"default": GROQ_ENDPOINT,
                     "tooltip": "OpenAI-compatible chat-completions endpoint."},
                ),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "prompt")
    FUNCTION = "generate"
    CATEGORY = "prompt/json"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return float(seed)

    def generate(
        self,
        model,
        user_prompt,
        schema_preset,
        seed,
        temperature,
        max_completion_tokens,
        reasoning_effort="medium",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        custom_schema="",
        api_key="",
        base_url=GROQ_ENDPOINT,
        top_p=1.0,
    ):
        key = api_key.strip() or os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "No Groq API key. Either fill the api_key widget or set the GROQ_API_KEY "
                "environment variable before launching ComfyUI."
            )

        # Resolve schema.
        if schema_preset == "custom":
            if not custom_schema.strip():
                raise ValueError("schema_preset is 'custom' but custom_schema is empty.")
            try:
                schema = json.loads(custom_schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"custom_schema is not valid JSON: {e}")
        else:
            schema = SCHEMA_PRESETS[schema_preset]

        strict = model in _STRICT_MODELS
        send_system_prompt = system_prompt
        if strict:
            state = {}
            gschema = _groqify_schema(schema, state)
            if state.get("bbox_rewritten"):
                # Keep the model's generation aligned with the rewritten (object) bbox, or
                # Groq's post-hoc schema validator returns a 400.
                send_system_prompt = system_prompt + _BBOX_OBJECT_OVERRIDE
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_composition",
                    "strict": True,
                    "schema": gschema,
                },
            }
        else:
            # Non-strict models can only guarantee valid JSON syntax, not the schema. The
            # system prompt carries the full shape; we validate keys after the fact.
            response_format = {"type": "json_object"}

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": send_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_completion_tokens": int(max_completion_tokens),
            "top_p": float(top_p),
            "seed": int(seed),
            "response_format": response_format,
            "stream": False,
        }
        # reasoning_effort is honored by the gpt-oss models; harmless to omit elsewhere.
        if model in _STRICT_MODELS:
            body["reasoning_effort"] = reasoning_effort

        req = urllib.request.Request(
            base_url.strip(),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Groq sits behind Cloudflare, which 403s (error 1010) on urllib's default
                # User-Agent. A normal UA gets through.
                "User-Agent": "ComfyUI-JSONPrompt/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"Groq API error {e.code}: {detail}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Groq ({base_url.strip()}): {e.reason}")

        try:
            raw = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Groq response shape: {json.dumps(payload)[:400]}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Groq output was not valid JSON ({e}). With a non-strict model this can "
                f"happen — pick gpt-oss-120b/20b for guaranteed structure, or raise "
                f"max_completion_tokens. First 200 chars: {raw[:200]!r}"
            )

        data = _bbox_objects_to_arrays(data)
        data = _normalize_hex_in_place(data)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return (pretty, _flatten_to_prompt(data))


NODE_CLASS_MAPPINGS = {"GroqJSONPromptGenerator": GroqJSONPromptGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"GroqJSONPromptGenerator": "Scene JSON Generator (Groq)"}
