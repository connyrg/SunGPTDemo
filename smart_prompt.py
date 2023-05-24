#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:45:43 2023

@author: conny
"""

from abc import ABC, abstractmethod

import re

INSTRUCTION_CONTEXT_QUESTION_PROMPT_TEMPLATE = """\
{instruction}

Context: {context}

Question: {question}

Answer:"""

WORD_TO_TOKEN_RATIO = 0.65

class Chain(ABC):
    
    @abstractmethod
    def generate(self, model, question, context):
        """LLM Q&A Chain"""
    

def _truncate_context(prompt_pattern, context, max_token_len, max_new_token_length):
    max_context_length = (max_token_len - max_new_token_length)*WORD_TO_TOKEN_RATIO - len(prompt_pattern.split())
    if (context=='') or (context is None):
        return context
    updated_context = re.match(fr"(^(?:\S+\s+){{,{int(max_context_length)-1}}}(?:\S+\s*))", context).group()
    return updated_context

class PDSCoverageChain(Chain):
    def generate(self, model, model_config, question, context):
        
        # Step 1 : Verified if the question about policy coverage
        instruction_hop1 = "Is the user's question about policy coverage? Respond with a yes or a no"
        context_hop1 = ''
        question_hop1 = question
        prompt_tmp = INSTRUCTION_CONTEXT_QUESTION_PROMPT_TEMPLATE.replace('{instruction}', instruction_hop1) \
                        .replace('{question}', question_hop1)
#        prompt = prompt_tmp.replace('{context}', context_hop1)
        updated_context = _truncate_context(prompt_tmp, context_hop1, 
                                           max_token_len=model_config['MAX_TOKENS'],
                                           max_new_token_length=model_config['MAX_NEW_TOKEN_LENGTH'])
        updated_prompt = prompt_tmp.replace('{context}', updated_context)
        
        response_hop1 = model.generate([updated_prompt]).generations[0][0].text.strip()
        print('step 1')
        print(updated_prompt)
        print(response_hop1)
        print('-----')
        
        # Validate answer
        answer = re.findall('(yes|no)', response_hop1, flags=re.IGNORECASE)
        if (len(answer)==0):
            answer = ''
        else:
            answer = answer[0]
        if answer.lower()!='yes':
            return response_hop1

        # Step 2 : Check policy coverage
        instruction_hop2 = "Answer the below questions based on the provided context. Be as detailed as possible."
        context_hop2 = context
        question_hop2 = "What is covered under the policy?"
        prompt_tmp = INSTRUCTION_CONTEXT_QUESTION_PROMPT_TEMPLATE.replace('{instruction}', instruction_hop2) \
                        .replace('{question}', question_hop2)
#        prompt = prompt_tmp.replace('{context}', context_hop2)
        updated_context = _truncate_context(prompt_tmp, context_hop2, 
                                           max_token_len=model_config['MAX_TOKENS'],
                                           max_new_token_length=model_config['MAX_NEW_TOKEN_LENGTH'])
        updated_prompt = prompt_tmp.replace('{context}', updated_context)
        
        response_hop2 = model.generate([updated_prompt]).generations[0][0].text.strip()
        print('step 2')
        print(updated_prompt)
        print(response_hop2)
        print('-----')
        
        # Step 3 : Check policy coverage
        instruction_hop3 = "Answer the below question based on the provided context. Think step by step and make an informed guess if the answer is not directly present in the context provided. Think step and step and explain your answer with reasons."
        context_hop3 = response_hop2
        question_hop3 = question
        prompt_tmp = INSTRUCTION_CONTEXT_QUESTION_PROMPT_TEMPLATE.replace('{instruction}', instruction_hop3) \
                        .replace('{question}', question_hop3)
#        prompt = prompt_tmp.replace('{context}', context_hop3)
        updated_context = _truncate_context(prompt_tmp, context_hop3, 
                                           max_token_len=model_config['MAX_TOKENS'],
                                           max_new_token_length=model_config['MAX_NEW_TOKEN_LENGTH'])
        updated_prompt = prompt_tmp.replace('{context}', updated_context)
        
        response_hop3 = model.generate([updated_prompt]).generations[0][0].text.strip()
        print('step 3')
        print(updated_prompt)
        print(response_hop3)
        print('-----')
        
        return response_hop3