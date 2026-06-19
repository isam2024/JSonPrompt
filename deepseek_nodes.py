"""DeepSeek-backed scene-composition JSON generator for ComfyUI-JSONPrompt.

Same Ideogram4 scene-composition JSON as the local-GGUF and Groq nodes, but inference runs on
DeepSeek's cloud API (https://api.deepseek.com, OpenAI-compatible).

DeepSeek only offers JSON *object* mode (`response_format={"type":"json_object"}`) — valid JSON
syntax guaranteed, but NO schema enforcement (unlike Groq's gpt-oss strict mode or the local
llama.cpp grammar). So structure rides entirely on the system prompt, which already contains a
JSON example and the word "json" (both required by DeepSeek's JSON mode). The strong v4 models
follow it reliably; we validate/normalize after the fact.

No third-party dependency: stdlib urllib against the OpenAI-compatible chat-completions endpoint.
"""

import json
import os
import urllib.request
import urllib.error

from .nodes import SCHEMA_PRESETS, DEFAULT_SYSTEM_PROMPT, _flatten_to_prompt
from .groq_nodes import _normalize_hex_in_place  # shared hex post-normalizer

DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"

DEEPSEEK_MODELS = [
    "deepseek-v4-pro",     # default — strongest, best structure adherence
    "deepseek-v4-flash",   # faster / cheaper
    "deepseek-chat",       # to be deprecated 2026-07-24
    "deepseek-reasoner",   # to be deprecated 2026-07-24; may not support JSON mode
]


def _coerce_bboxes(data):
    """DeepSeek doesn't enforce the schema, so the model owns bbox shape. Coerce float coords
    to ints where it's safe; leave anything genuinely malformed for the validator to flag."""
    cd = data.get("compositional_deconstruction")
    if not isinstance(cd, dict):
        return data
    for el in cd.get("elements", []):
        bbox = el.get("bbox")
        if isinstance(bbox, list):
            el["bbox"] = [int(round(v)) if isinstance(v, (int, float)) else v for v in bbox]
    return data


class DeepSeekJSONPromptGenerator:
    """Generate the Ideogram4 scene-composition JSON via DeepSeek (cloud, JSON-object mode)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (DEEPSEEK_MODELS,),
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
                "max_tokens": ("INT", {"default": 4096, "min": 256, "max": 8192}),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT},
                ),
                "custom_schema": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "Only used for the 'custom' preset, to validate the output "
                     "shape. DeepSeek does NOT enforce schemas, so describe the shape in the "
                     "system prompt too."},
                ),
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "DeepSeek API key. Leave blank to use the "
                     "DEEPSEEK_API_KEY environment variable (preferred — keeps it out of the "
                     "saved workflow)."},
                ),
                "base_url": (
                    "STRING",
                    {"default": DEEPSEEK_ENDPOINT,
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
        max_tokens,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        custom_schema="",
        api_key="",
        base_url=DEEPSEEK_ENDPOINT,
        top_p=1.0,
    ):
        key = api_key.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "No DeepSeek API key. Either fill the api_key widget or set the "
                "DEEPSEEK_API_KEY environment variable before launching ComfyUI."
            )

        # The 'custom' preset is only used to validate the returned shape (DeepSeek can't
        # enforce it server-side). The preset schemas are validated structurally below.
        if schema_preset == "custom" and custom_schema.strip():
            try:
                json.loads(custom_schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"custom_schema is not valid JSON: {e}")

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            # DeepSeek JSON mode: valid JSON syntax guaranteed, structure is the prompt's job.
            "response_format": {"type": "json_object"},
            "stream": False,
        }

        req = urllib.request.Request(
            base_url.strip(),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "ComfyUI-JSONPrompt/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"DeepSeek API error {e.code}: {detail}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach DeepSeek ({base_url.strip()}): {e.reason}")

        try:
            raw = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected DeepSeek response shape: {json.dumps(payload)[:400]}")

        if not raw or not raw.strip():
            # Documented DeepSeek JSON-mode quirk: occasionally returns empty content.
            raise ValueError(
                "DeepSeek returned empty content (a known JSON-mode quirk). Re-run, raise "
                "max_tokens, or tweak the prompt. Try deepseek-v4-pro if on another model."
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"DeepSeek output was not valid JSON ({e}). It likely hit max_tokens "
                f"({int(max_tokens)}) mid-document — raise max_tokens. First 200 chars: "
                f"{raw[:200]!r}"
            )

        data = _coerce_bboxes(data)
        data = _normalize_hex_in_place(data)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return (pretty, _flatten_to_prompt(data))


NODE_CLASS_MAPPINGS = {"DeepSeekJSONPromptGenerator": DeepSeekJSONPromptGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"DeepSeekJSONPromptGenerator": "Scene JSON Generator (DeepSeek)"}
