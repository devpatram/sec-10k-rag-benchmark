"""A local chat model from Hugging Face, called with greedy decoding so a run can be repeated."""
from __future__ import annotations


class ChatModel:
    def __init__(self, name: str):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_arg = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in transformers 4.56
        kwargs = {dtype_arg: dtype}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def __call__(self, prompt: str, max_new_tokens: int = 160) -> str:
        import torch

        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
