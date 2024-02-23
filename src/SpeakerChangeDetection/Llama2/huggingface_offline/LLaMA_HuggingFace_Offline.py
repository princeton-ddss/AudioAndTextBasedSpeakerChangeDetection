'''
Run LLaMA Offline under HuggingFace for the Speaker Segmentation and Speaker Diarization
'''

#!/usr/bin/env python
# coding: utf-8
import os
import torch
import gc
from huggingface_hub import snapshot_download
import transformers
from transformers import AutoModelForCausalLM

def get_full_prompt(systemprompt,  main_question, questions_answers_dict = None,
                    INST = '[INST]', E_INST = '[/INST]',
                    SYS = '<<SYS>>\n', E_SYS = '\n<</SYS>>\n\n',
                    SEN = '<s>', EN_SEN = '<\s>'):
    if not questions_answers_dict: # No few-shot learning
        full_prompt =   SEN + INST + \
                        SYS + systemprompt + E_SYS + \
                        main_question + E_INST + EN_SEN
    else:
        questions = list(questions_answers_dict.keys())
        full_prompt =   SEN + INST + \
                        SYS + systemprompt + E_SYS + \
                        questions[0] + E_INST + \
                        questions_answers_dict[questions[0]] + EN_SEN

        for question in questions[1:]:
            full_prompt += SEN + INST + question + E_INST + questions_answers_dict[question] + EN_SEN

        full_prompt += SEN + INST + main_question + E_INST
    return full_prompt

def setup_LLaMA_model_tokenizer(llama_model_dir, llama_folder_name, device_map = '0', torch_dtype=torch.float16):
    # By default: Set to use only one GPU Device 0
    # Use GPU Device 0 to use the model
    if device_map != 'auto':
        os.environ['CUDA_VISIBLE_DEVICES'] = device_map
    llama_model = AutoModelForCausalLM.from_pretrained(os.path.join(llama_model_dir, llama_folder_name),
                                                       trust_remote_code=True, device_map=device_map, torch_dtype=torch_dtype)

    # Set up Tokenizer and Pipeline
    # Import Tokenizer and Pipeline
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(llama_model_dir, llama_folder_name),
                                                           return_token_type_ids=False)
    pipeline = transformers.pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=tokenizer
    )
    return tokenizer, pipeline








