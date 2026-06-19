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

## Cloud backend (Groq) — "Scene JSON Generator (Groq)"

A second node, `GroqJSONPromptGenerator`, produces the **same scene-composition JSON** but runs
inference on [Groq](https://console.groq.com) instead of a local GGUF. No VRAM cost, no
llama.cpp build — useful when the GPU is busy with the diffusion model.

- Default model `openai/gpt-oss-120b` (also `-20b`) uses Groq **strict structured outputs**, so
  the key set / types / `type:"obj"` enum are schema-enforced server-side. Other models
  (Kimi, Llama-3.3, Qwen) fall back to JSON-object mode (valid JSON, no schema enforcement).
- **API key:** leave the `api_key` widget blank and set the `GROQ_API_KEY` environment variable
  (preferred — keeps the key out of saved workflows), or paste it into the widget.
- Output is byte-for-byte interchangeable with the local node (same JSON shape, `bbox` as a
  4-int `[y_min, x_min, y_max, x_max]` array — Ideogram's canonical y-first order, which the
  KJNodes Prompt Builder `import_json` expects — and uppercase `#RRGGBB`).

Implementation notes (Groq strict mode differs from llama.cpp grammar):
- Groq strict mode only honors `type/properties/items/required/additionalProperties/enum`. It
  **ignores** `maxLength`/`pattern`/`minItems`/`maxItems`, so those are stripped from the request;
  string-length discipline is left to the system prompt, and hex is normalized in post.
- A fixed-length array (the 4-int `bbox`) can't be enforced that way — the model emits degenerate
  bboxes (`[1,1]`, `[100500900800]`). The node sends `bbox` as a **4-required-field object**
  (`x_min/y_min/x_max/y_max`, which strict mode *does* guarantee), appends a matching bbox-format
  instruction to the system prompt, and converts the object back to the canonical array on output.
- Posts to the OpenAI-compatible endpoint with stdlib `urllib` (no `groq` SDK dependency); sends a
  `User-Agent` header because Groq's Cloudflare front 403s urllib's default (error 1010).

## Install

1. Copy this `ComfyUI-JSONPrompt/` folder into `ComfyUI/custom_nodes/`.
2. Install the engine into ComfyUI's Python environment:
   ```
   <ComfyUI python> -m pip install -r requirements.txt
   ```
   CPU works fine for a small model. For NVIDIA GPU offload see **GPU acceleration** below
   (note: on consumer Intel 12th–14th gen CPUs you generally must *build* the CUDA wheel
   from source — prebuilt CUDA wheels can crash). On Mac, a Metal wheel builds automatically.
3. Put a GGUF model in `ComfyUI/models/llm_gguf/`. Good options:
   - Your existing **Gemma** GGUF (best quality, heavier).
   - **Qwen2.5-1.5B-Instruct** Q4_K_M (~1 GB) — fast, surprisingly good with a grammar.
   - **Qwen2.5-0.5B-Instruct** Q4_K_M (~400 MB) — fastest; structure still perfect, content simpler.
4. Restart ComfyUI. The node appears as **Scene JSON Generator (local LLM)** under `prompt/json`.

## Node I/O

**Inputs:** `model_name` (dropdown of GGUFs), `user_prompt` (the image request),
`schema_preset` (`scene_composition` or `custom`), `seed`, `temperature`, `max_tokens`,
plus optional `system_prompt` (defaults to your exact prompt), `custom_schema`,
`model_path_override`, `n_ctx`, `n_threads`, `n_gpu_layers`
(`-1` = offload all layers to GPU, needs a CUDA build; `0` = CPU only).

**Outputs:** `json` (pretty-printed, guaranteed-valid) and `prompt` (a flattened comma string
of the descriptive fields, for convenience — ignore it if you consume the JSON directly).

## GPU acceleration

Set `n_gpu_layers = -1` to offload the whole model to an NVIDIA GPU. This requires a
**CUDA-enabled build** of `llama-cpp-python`; a CPU-only build silently ignores `n_gpu_layers`
and runs on CPU.

⚠️ **Prebuilt CUDA wheels can crash on consumer Intel CPUs.** Wheels from the public indexes
(abetlen, community builds) are compiled with **AVX-512** baked in. Intel removed AVX-512 from
12th–14th gen consumer chips (Alder/Raptor Lake), so on those CPUs the model loads and offloads
fine but then dies at context creation with `OSError [WinError -1073741795] (0xc000001d,
STATUS_ILLEGAL_INSTRUCTION)`. There is no runtime flag to disable it — the instructions are
compiled in.

**Fix: build from source**, so llama.cpp's `GGML_NATIVE` detects your CPU and emits only the
SIMD it supports (AVX2, not AVX-512). On Windows you need VS Build Tools (MSVC), the CUDA
Toolkit (nvcc), and ninja:

```bat
:: from a shell with the venv on PATH
pip install ninja
call "...\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_GENERATOR=Ninja
set CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
set FORCE_CMAKE=1
<venv python> -m pip install "llama-cpp-python" --no-binary llama-cpp-python ^
    --force-reinstall --no-deps --no-cache-dir
```

Notes: use **Ninja** (the default VS generator fails with `No CUDA toolset found` unless CUDA's
MSBuild integration is installed into the VS instance). Set `CMAKE_CUDA_ARCHITECTURES` to your
GPU's compute capability (`89` = Ada / RTX 40-series; `86` = Ampere; `75` = Turing) to keep the
compile short. Verify with `python -c "from llama_cpp import llama_cpp;
print(llama_cpp.llama_supports_gpu_offload())"` and look for `offloaded N/N layers to GPU` in the
load log. On Linux a prebuilt CUDA wheel from the abetlen index usually works without the AVX-512
problem; this gotcha is specific to AVX-512-built Windows wheels on consumer Intel CPUs.

## Notes

- The model loads **once** and is cached; later generations pay inference cost only.
- Switching `model_name`, `n_ctx`, or `n_threads` reloads (and frees the previous model).
- Bump `max_tokens` if scenes with many elements get truncated; the grammar will still keep
  partial output valid up to the limit, but you want enough room to close the document.
