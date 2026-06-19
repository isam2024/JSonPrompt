"""Core node implementation for ComfyUI-JSONPrompt.

Generates the scene-composition JSON with a local GGUF model, using a JSON-schema-derived
grammar so the output is structurally guaranteed to match the spec (exact keys, hex palettes,
4-int bboxes, no prose, no fences). The model is responsible only for content quality.
"""

import json
import os

# llama-cpp-python is the inference engine. Imported lazily so ComfyUI can still load the
# package (and show a clear error) if the dependency is missing.
try:
    from llama_cpp import Llama
    _LLAMA_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - environment dependent
    Llama = None
    _LLAMA_IMPORT_ERROR = e

# ComfyUI's folder_paths lets us discover GGUF files dropped in models/llm_gguf/.
try:
    import folder_paths

    _GGUF_DIR = os.path.join(folder_paths.models_dir, "llm_gguf")
    os.makedirs(_GGUF_DIR, exist_ok=True)
    if "llm_gguf" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["llm_gguf"] = ([_GGUF_DIR], {".gguf"})
except Exception:
    folder_paths = None
    _GGUF_DIR = os.path.join(os.path.dirname(__file__), "models")


# ---------------------------------------------------------------------------
# Model cache: load the GGUF once, reuse across generations. Per-generation
# cost becomes inference only, never load.
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


def _get_model(model_path, n_ctx, n_threads, n_gpu_layers):
    key = (os.path.abspath(model_path), int(n_ctx), int(n_threads), int(n_gpu_layers))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it into ComfyUI's Python:\n"
            "    pip install llama-cpp-python\n"
            f"(original import error: {_LLAMA_IMPORT_ERROR})"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF model not found: {model_path}")
    _MODEL_CACHE.clear()  # keep only one model resident
    # n_gpu_layers=-1 offloads every layer to the GPU. Requires a CUDA-enabled build of
    # llama-cpp-python; a CPU-only build silently ignores this and stays on CPU.
    llm = Llama(
        model_path=model_path,
        n_ctx=int(n_ctx),
        n_threads=int(n_threads) if int(n_threads) > 0 else None,
        n_gpu_layers=int(n_gpu_layers),
        verbose=True,  # prints "offloaded X/Y layers to GPU" so you can confirm GPU use
    )
    _MODEL_CACHE[key] = llm
    return llm


# ---------------------------------------------------------------------------
# Schemas. The scene-composition schema mirrors the existing Gemma spec exactly,
# but expressed so llama.cpp compiles it to an enforcing grammar.
# ---------------------------------------------------------------------------
_HEX = {"type": "string", "pattern": "^#[0-9A-F]{6}$"}  # uppercase enforced by the grammar

SCENE_COMPOSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "high_level_description": {"type": "string", "maxLength": 300},
        "style_description": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "aesthetics": {"type": "string", "maxLength": 180},
                "lighting": {"type": "string", "maxLength": 180},
                "photo": {"type": "string", "maxLength": 180},
                "medium": {"type": "string", "maxLength": 50},
                "color_palette": {"type": "array", "items": _HEX, "minItems": 3, "maxItems": 6},
            },
            "required": ["aesthetics", "lighting", "photo", "medium", "color_palette"],
        },
        "compositional_deconstruction": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "background": {"type": "string", "maxLength": 350},
                "elements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["obj"]},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "desc": {"type": "string", "maxLength": 300},
                            "color_palette": {
                                "type": "array",
                                "items": _HEX,
                                "minItems": 2,
                                "maxItems": 5,
                            },
                        },
                        "required": ["type", "bbox", "desc", "color_palette"],
                    },
                },
            },
            "required": ["background", "elements"],
        },
    },
    "required": [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ],
}

SCHEMA_PRESETS = {"scene_composition": SCENE_COMPOSITION_SCHEMA}


# The existing system prompt, verbatim. Editable in the node so you can tune wording
# without touching code. The grammar enforces structure regardless of what this says.
DEFAULT_SYSTEM_PROMPT = """You are a scene composition assistant. Given a user request for an image, you output a single JSON document that describes the scene in a structured, render-ready form. You output JSON only — no prose, no markdown fences, no commentary.

# Output format

Your response MUST be a single valid JSON object matching exactly this shape and key set:

```
{
  "high_level_description": "",
  "style_description": {
    "aesthetics": "",
    "lighting": "",
    "photo": "",
    "medium": "",
    "color_palette": []
  },
  "compositional_deconstruction": {
    "background": "",
    "elements": [
      {
        "type": "obj",
        "bbox": [0, 0, 0, 0],
        "desc": "",
        "color_palette": []
      }
    ]
  }
}
```

All keys above are required and must appear exactly as named. Do not add, rename, or remove any keys.

# Field rules

## high_level_description

- String, **50-word hard cap**. ONE long sentence preferred, never more than two. Start immediately with the subject — no "this image shows", "depicts", "captures". Identify the main subject(s), medium, and overall composition. General terms (`various`, `multiple`) are fine here; granular detail belongs in element `desc`s and `background`.
- **Name** the subjects and the scene, but do NOT spell out their specific attributes (clothing, colors, pose, held objects, materials) here — those belong ONLY in the matching element. Repeating a subject's concrete details here AND in an element makes the renderer draw that subject TWICE.

## style_description

A flat object describing how the image is rendered, independent of what it depicts.

- `aesthetics` (string): Overall visual style and treatment.
- `lighting` (string): Light source, direction, quality, and color temperature.
- `photo` (string): Camera/lens/photographic specifics when relevant. Use an empty string "" if the medium is not photographic.
- `medium` (string): The medium category (e.g. "photography", "oil painting", "3D render", "watercolor", "digital illustration").
- `color_palette` (array of strings): 3–6 dominant colors of the overall image as uppercase hex codes in #RRGGBB form.

## compositional_deconstruction.background

- String, **60-word cap**. Describe only the environment behind and around the subjects, plus scene-wide lighting/atmosphere and any shadows. Do NOT describe any element listed in `elements`.

## compositional_deconstruction.elements

Array with at least 1 item, listed roughly background-to-foreground.

Each element:

- `type` (string): Always "obj".
- `bbox` (array of 4 integers): **[y_min, x_min, y_max, x_max]** on a 1000×1000 canvas with origin at the top-left, y increasing downward, x increasing rightward. This is Ideogram's canonical row-major order — **y comes FIRST**. Must satisfy 0 ≤ y_min < y_max ≤ 1000 and 0 ≤ x_min < x_max ≤ 1000. The box must reflect the element's described position and relative size.
- `desc` (string): **30–60 words, 60-word HARD CAP.** Identity FIRST (a standalone catalog entry — open with what the thing is, not "the X"), then major attributes briefly (people: skin tone, hair, each garment + color, expression, pose; objects: shape, material, color, distinctive parts), then one distinguishing detail. **One subject = one element** — anatomical/structural parts go in that element's desc, never as separate elements. Do NOT include: camera/render language (DoF, bokeh, focus, grain, lens flare) unless the user asked; shadow language (the renderer infers shadows — scene-wide ones go in `background`); metaphor/impression words (luminous, radiant, vibrant, lush, stunning, breathtaking) — use observable properties instead. Do not restate global background or style information.
- `color_palette` (array of strings): 2–5 dominant colors of THIS element as uppercase hex codes in #RRGGBB form.

# Composition guidance

- **One entity = one element, described once.** Every distinct subject/object appears as exactly ONE element and is described in detail in that element ONLY. Never create two elements for the same entity, and never re-describe an element's subject in `high_level_description` or `background`. This is the #1 cause of unwanted duplicate subjects in the render.
- Place elements deliberately: vary depth, avoid centering everything, and let bboxes match the prose.
- Keep `style_description` and every `desc` mutually consistent in palette, lighting, and atmosphere.
- Each element's `color_palette` should be plausibly drawn from or harmonious with the overall `style_description.color_palette`.
- Prefer 3–8 elements unless the user explicitly asks for more or fewer.

# Hard constraints

- Output valid JSON and nothing else.
- Use only the keys defined above, exactly as spelled. No extra fields.
- Do not wrap the JSON in code fences or add explanations.

# Instruction

Generate the JSON based on this user prompt:"""


# Keys whose values are structural noise for the optional flattened-prompt output.
_SKIP_FLATTEN_KEYS = {"bbox", "type", "color_palette"}


def _flatten_to_prompt(data):
    """Optional convenience: a comma-joined positive prompt from the descriptive strings.

    Skips structural noise (bboxes, hex palettes, the literal "obj"). The primary output
    is the JSON; this is just a ready-to-encode fallback string.
    """
    parts = []

    def collect(value, key=None):
        if key in _SKIP_FLATTEN_KEYS:
            return
        if isinstance(value, str):
            v = value.strip()
            if v and not v.startswith("#"):
                parts.append(v)
        elif isinstance(value, list):
            for item in value:
                collect(item, key)
        elif isinstance(value, dict):
            for k, v in value.items():
                collect(v, k)

    collect(data)
    return ", ".join(parts)


class JSONPromptGenerator:
    """Generate the scene-composition JSON with a local GGUF model (grammar-enforced)."""

    @classmethod
    def INPUT_TYPES(cls):
        if folder_paths is not None:
            try:
                gguf_files = folder_paths.get_filename_list("llm_gguf")
            except Exception:
                gguf_files = []
        else:
            gguf_files = []
        model_choices = gguf_files if gguf_files else ["<put a .gguf in models/llm_gguf>"]

        return {
            "required": {
                "model_name": (model_choices,),
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
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192}),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT},
                ),
                "custom_schema": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "JSON Schema used when schema_preset = custom"},
                ),
                "model_path_override": (
                    "STRING",
                    {"default": "", "tooltip": "Absolute path to a .gguf (overrides model_name)"},
                ),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 128}),
                "n_gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999,
                     "tooltip": "-1 = offload all layers to GPU (needs CUDA build). 0 = CPU only."},
                ),
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
        model_name,
        user_prompt,
        schema_preset,
        seed,
        temperature,
        max_tokens,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        custom_schema="",
        model_path_override="",
        n_ctx=4096,
        n_threads=0,
        n_gpu_layers=-1,
    ):
        # Resolve model path.
        if model_path_override.strip():
            model_path = model_path_override.strip()
        elif folder_paths is not None:
            model_path = folder_paths.get_full_path("llm_gguf", model_name)
            if not model_path:
                raise FileNotFoundError(
                    f"'{model_name}' not found in models/llm_gguf. "
                    "Drop a GGUF there (e.g. your Gemma GGUF, or a smaller instruct model)."
                )
        else:
            model_path = os.path.join(_GGUF_DIR, model_name)

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

        llm = _get_model(model_path, n_ctx, n_threads, n_gpu_layers)

        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # The schema is compiled to a grammar: structure, key set, enums, hex pattern,
            # and bbox arity are all enforced at the token level. Malformed JSON is impossible.
            response_format={"type": "json_object", "schema": schema},
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            seed=int(seed),
        )

        raw = result["choices"][0]["message"]["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            # The schema bounds every string (maxLength) and the element count (maxItems),
            # so the grammar forces a complete document — this should not happen. If it does,
            # the generation hit max_tokens before closing. Fail with a clear, actionable message.
            raise ValueError(
                f"Model output was not valid JSON ({e}). It likely hit max_tokens "
                f"({int(max_tokens)}) before the document closed — raise max_tokens or "
                f"lower temperature. First 200 chars: {raw[:200]!r}"
            )
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return (pretty, _flatten_to_prompt(data))


NODE_CLASS_MAPPINGS = {"JSONPromptGenerator": JSONPromptGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"JSONPromptGenerator": "Scene JSON Generator (local LLM)"}
