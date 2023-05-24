#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:29:57 2023

@author: conny
"""


from transformers import pipeline

def get_model_pipeline(model_name):
    #generator = pipeline(model="gpt2")
    if model_name=='TheBloke/stable-vicuna-13B-HF':
        return pipeline(model="TheBloke/stable-vicuna-13B-HF", task='text-generation')
    
    raise ValueError(f'Unrecognised model_name {model_name}')

def get_inference(model_pipeline, prompts):
    outputs = model_pipeline(prompts, return_full_text=False)
    return outputs
    