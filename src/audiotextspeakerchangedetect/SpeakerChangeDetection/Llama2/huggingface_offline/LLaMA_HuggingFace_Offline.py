"""
Run LLaMA Offline under HuggingFace for the Speaker Segmentation and Speaker Diarization
"""

#!/usr/bin/env python
# coding: utf-8
import os
import torch
import transformers
from transformers import AutoModelForCausalLM


def setup_llama_tokenizer(
    model_dir: str,
    device_map="0",
    torch_dtype=torch.float16
):
    """..."""

    # Use a single GPU (device=0) by default
    if device_map != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, return_token_type_ids=False
    )
    pipeline = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer
    )

    return tokenizer, pipeline
