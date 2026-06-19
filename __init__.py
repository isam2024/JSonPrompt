"""ComfyUI-JSONPrompt — generate guaranteed-valid JSON image prompts with a tiny local LLM.

A small GGUF model (e.g. Qwen2.5-0.5B-Instruct) is run through llama-cpp-python with a
JSON-schema-derived grammar, so the output can never be malformed JSON. The model is loaded
once and cached, so per-generation cost is inference only.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Groq (cloud) backend — optional. Registered in a guard so a failure here can never stop the
# local node from loading.
try:
    from .groq_nodes import (
        NODE_CLASS_MAPPINGS as _GROQ_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _GROQ_NAMES,
    )
    NODE_CLASS_MAPPINGS.update(_GROQ_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_GROQ_NAMES)
except Exception as e:  # pragma: no cover
    print(f"[ComfyUI-JSONPrompt] Groq node not loaded: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
