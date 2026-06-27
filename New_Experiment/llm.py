"""
llm.py — local inference wrapper with vision-language model support.

"""

from __future__ import annotations

import os

# Must be set before any HuggingFace import
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))

try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib = None


def _is_qwen_vl(model_name: str) -> bool:
    return "Qwen2.5-VL" in model_name or "Qwen2-VL" in model_name


class LocalLLM:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        top_p: float = 0.95,
        seed: int | None = None,
    ):
        self.model_name     = model_name
        self.temperature    = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p          = top_p
        self.seed           = seed
        self.is_ollama      = "/" not in model_name
        self._is_vl         = _is_qwen_vl(model_name)

        if self.is_ollama:
            if _ollama_lib is None:
                raise ImportError("pip install ollama")
            print(f"[LocalLLM] using Ollama: {model_name}", flush=True)

        elif self._is_vl:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self._torch = torch
            if seed is not None:
                torch.manual_seed(seed)

            dtype      = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto"         if torch.cuda.is_available() else None

            print(f"[LocalLLM] VL model detected: {model_name}", flush=True)
            print(f"[LocalLLM] loading processor: {model_name}", flush=True)
            self.processor = AutoProcessor.from_pretrained(
                model_name, local_files_only=True
            )

            print(f"[LocalLLM] loading model: {model_name} ({dtype})", flush=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                local_files_only=True,
            )
            self.model.eval()
            print("[LocalLLM] ready", flush=True)

        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            if seed is not None:
                torch.manual_seed(seed)

            dtype      = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto"         if torch.cuda.is_available() else None

            print(f"[LocalLLM] loading tokenizer: {model_name}", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            print(f"[LocalLLM] loading model: {model_name} ({dtype})", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                local_files_only=True,
            )
            self.model.eval()
            print("[LocalLLM] ready", flush=True)

    # -----------------------------------------------------------------------

    def generate(self, messages: list[dict]) -> str:
        if self.is_ollama:
            response = _ollama_lib.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p":       self.top_p,
                    "num_predict": self.max_new_tokens,
                    "seed":        self.seed,
                },
            )
            return response["message"]["content"]

        if self._is_vl:
            return self._generate_vl(messages)

        return self._generate_causal(messages)

    # -----------------------------------------------------------------------

    def _generate_vl(self, messages: list[dict]) -> str:
        """Text-only generation for Qwen2.5-VL using AutoProcessor."""
        torch = self._torch
        text  = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        generated_ids = [
            out[len(inp):]
            for out, inp in zip(output_ids, inputs["input_ids"])
        ]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _generate_causal(self, messages: list[dict]) -> str:
        """Standard generation for AutoModelForCausalLM models."""
        torch       = self._torch
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
