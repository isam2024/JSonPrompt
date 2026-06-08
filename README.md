# ComfyUI-JSONPrompt

A ComfyUI node that generates your **scene-composition JSON** with a small, local GGUF model —
and a JSON-schema-derived **grammar** so the output is *structurally guaranteed* to match the
spec. No more "did the model produce valid JSON this time?" The model only decides *content*;
the structure cannot break.

## Why this exists

Running a 4B+ LLM just to get reliable JSON is heavy, and the small/fast models are the ones
that botch the structure. Grammar-constrained decoding fixes this: at every step the engine
masks out any token that would violate the schema, so even a tiny model produces perfect JSON.
You can keep your current Gemma GGUF for quality, or drop to a much smaller/faster model — the
JSON guarantee is identical either way.

What the grammar enforces for free (was previously "please obey" in the prompt):
- valid JSON, no prose, no markdown fences
- exactly the spec's key set, nothing added/renamed (`additionalProperties: false`)
- `type` is always `"obj"` (enum)
- palette entries are uppercase `#RRGGBB` (regex pattern)
- `bbox` is exactly 4 integers; `elements` has ≥1 item

What the *model* still owns (semantic, not structural — no engine guarantees these):
`x_min < x_max`, bboxes matching the prose, palette harmony, description quality.

## Install

1. Copy this `ComfyUI-JSONPrompt/` folder into `ComfyUI/custom_nodes/`.
2. Install the engine into ComfyUI's Python environment:
   ```
   <ComfyUI python> -m pip install -r requirements.txt
   ```
   (On Mac, a Metal-enabled wheel is built automatically. On NVIDIA, install a CUDA wheel
   of llama-cpp-python if you want GPU offload — CPU works fine for a small model.)
3. Put a GGUF model in `ComfyUI/models/llm_gguf/`. Good options:
   - Your existing **Gemma** GGUF (best quality, heavier).
   - **Qwen2.5-1.5B-Instruct** Q4_K_M (~1 GB) — fast, surprisingly good with a grammar.
   - **Qwen2.5-0.5B-Instruct** Q4_K_M (~400 MB) — fastest; structure still perfect, content simpler.
4. Restart ComfyUI. The node appears as **Scene JSON Generator (local LLM)** under `prompt/json`.

## Node I/O

**Inputs:** `model_name` (dropdown of GGUFs), `user_prompt` (the image request),
`schema_preset` (`scene_composition` or `custom`), `seed`, `temperature`, `max_tokens`,
plus optional `system_prompt` (defaults to your exact prompt), `custom_schema`,
`model_path_override`, `n_ctx`, `n_threads`.

**Outputs:** `json` (pretty-printed, guaranteed-valid) and `prompt` (a flattened comma string
of the descriptive fields, for convenience — ignore it if you consume the JSON directly).

## Notes

- The model loads **once** and is cached; later generations pay inference cost only.
- Switching `model_name`, `n_ctx`, or `n_threads` reloads (and frees the previous model).
- Bump `max_tokens` if scenes with many elements get truncated; the grammar will still keep
  partial output valid up to the limit, but you want enough room to close the document.
