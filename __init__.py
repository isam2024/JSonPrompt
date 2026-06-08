"""ComfyUI-JSONPrompt — generate guaranteed-valid JSON image prompts with a tiny local LLM.

A small GGUF model (e.g. Qwen2.5-0.5B-Instruct) is run through llama-cpp-python with a
JSON-schema-derived grammar, so the output can never be malformed JSON. The model is loaded
once and cached, so per-generation cost is inference only.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
