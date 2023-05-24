import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#import torch
#print(f"Is CUDA available: {torch.cuda.is_available()}")
## True
#print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
## Tesla T4


import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from text_generation import InferenceAPIClient
#from langchain import HuggingFaceHub

from langchain.llms import OpenAI
#from langchain.chat_models import ChatOpenAI

#from langchain.chains import RetrievalQA

#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import json
import pandas as pd
import re

import model as model_pipeline
from smart_prompt import PDSCoverageChain

st.set_page_config(layout="wide")

# Parameters
TITLE = 'SunGPT Demo App'
MODEL_OPTIONS = [
    "OpenAI/GPT-4",
    "OpenAI/GPT-3.5-turbo",
    "OpenAssistant/SFT-4 Pythia 12B",
#    "TheBloke/stable-vicuna-13B-HF",
]
EMBEDDING_OPTIONS = [
    'OpenAIEmbeddings',
#    'InstructEmbeddings-Large',
#    'InstructEmbeddings-XL',
    ]
STRATEGY_OPTIONS = [
    'Chain-of-Thought',
    'Without Chain-of-Thought',
]

@st.cache_data
def load_pdf_document(file_path='https://www.aami.com.au/aami/documents/personal/home/aami-home-building-insurance-pds.pdf'):
    print('Load documents....')
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

@st.cache_resource
def init_embeddings():
    """
    Supported embeddings:
        hkunlp/instructor-large
        hkunlp/instructor-xl
        OpenAI
    """
    print('Initializing embeddings....')
    embeddings = {}
    _ = embeddings.update({
        'OpenAIEmbeddings': OpenAIEmbeddings()
    })
#    _ = embeddings.update({
#        'InstructEmbeddings-Large': HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
#    })
#    _ = embeddings.update({
#        'InstructEmbeddings-XL': HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
#    })
    
    return embeddings

@st.cache_resource
def init_models():
    print('Initializing models....')
    models = {}
    model_configs = {}
    
    # OpenAI
    _ = model_configs.update({
        'OpenAI/GPT-4':{
            'MAX_TOKENS': 8192, 
            'MAX_NEW_TOKEN_LENGTH': 512,
            'USER_TOKEN': '',
            'END_TOKEN': '',
            'ASSISTANT_TOKEN': '',
        }})
    model = OpenAI(
        model_name='gpt-4',
        max_tokens=model_configs['OpenAI/GPT-4']['MAX_NEW_TOKEN_LENGTH'],
        temperature=0.001,
        model_kwargs= {'top_p':0.95},
    )
    _ = models.update({
        'OpenAI/GPT-4': model
    })
    
    _ = model_configs.update({
        'OpenAI/GPT-3.5-turbo':{
            'MAX_TOKENS': 4096, 
            'MAX_NEW_TOKEN_LENGTH': 512,
            'USER_TOKEN': '',
            'END_TOKEN': '',
            'ASSISTANT_TOKEN': '',
        }})
    model = OpenAI(
        model_name='gpt-3.5-turbo',
        max_tokens=model_configs['OpenAI/GPT-3.5-turbo']['MAX_NEW_TOKEN_LENGTH'],
        temperature=0.001,
        model_kwargs= {'top_p':0.95},
    )
    _ = models.update({
        'OpenAI/GPT-3.5-turbo': model
    })

    
    # Pythia
    MODEL_NAME = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'
    _ = model_configs.update({
        'OpenAssistant/SFT-4 Pythia 12B': {
            'MAX_TOKENS': 2048, 
            'MAX_NEW_TOKEN_LENGTH': 256,
            'USER_TOKEN': '<|prompter|>',
            'END_TOKEN': '<|endoftext|>',
            'ASSISTANT_TOKEN': '<|assistant|>',
        }})
    client = InferenceAPIClient(MODEL_NAME, token=os.getenv("HF_TOKEN"))
    model = HuggingFaceTextGenInference(
        inference_server_url='https://api-inference.huggingface.co/models/'+MODEL_NAME,
        max_new_tokens=model_configs['OpenAssistant/SFT-4 Pythia 12B']['MAX_NEW_TOKEN_LENGTH'],
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.001,
        repetition_penalty=1.03,
        client=client,
    )
    _ = models.update({
        'OpenAssistant/SFT-4 Pythia 12B': model
    }),

#    MODEL_NAME = 'TheBloke/stable-vicuna-13B-HF'
#    model = HuggingFaceHub(repo_id=MODEL_NAME, model_kwargs={"temperature":0})
#    model = model_pipeline.get_model_pipeline(MODEL_NAME)
#    models.update({
#        'TheBloke/stable-vicuna-13B-HF': model
#    }),
    
    return models, model_configs

def init_app():
    embeddings = init_embeddings()
    models, model_configs = init_models()
    return models, model_configs, embeddings

def get_retriever(texts, embeddings):
    print('get_retriever')
    retriever = Chroma.from_documents(texts, embeddings) \
                    .as_retriever(search_type='similarity',search_kwargs={"k":3})
    return retriever

def get_retriever_from_text(texts, embeddings):
    print('get_retriever')
    retriever = Chroma.from_texts(texts, embeddings) \
                    .as_retriever(search_type='similarity',search_kwargs={"k":3})
    return retriever

def get_special_tokens(model):
    if model in ['OpenAssistant/SFT-4 Pythia 12B']:
        user_token = '<|prompter|>'
        end_token = '<|endoftext|>'
        assistant_token = '<|assistant|>'
        return user_token, end_token, assistant_token
    if model in ['TheBloke/stable-vicuna-13B-HF']:
        user_token = '### Human: '
        end_token = ''
        assistant_token = '### Assistant:'
        return user_token, end_token, assistant_token
    else:
        return '', '', ''

def convert_transcript(transcript_json):
    text = '\n\n'.join(pd.DataFrame(transcript_json)[['sp','w']].apply(lambda x: x.sp+': '+x.w, axis='columns'))
    return text

def truncate_context(prompt_pattern, context, max_token_len, max_new_token_length):
    max_context_length = (max_token_len - max_new_token_length)*0.65 - len(prompt_pattern.split())
    updated_context = re.match(fr"(^(?:\S+\s+){{,{int(max_context_length)-1}}}(?:\S+\s*))", context).group()
    return updated_context
    
def reset_session_state():
    st.session_state['selected_embeddings'] = EMBEDDING_OPTIONS[0]
    st.session_state['pdf_file'] = None
    st.session_state['retriever'] = None
    
# Model initialization
models, model_configs, embeddings = init_app()

# Session state
if 'selected_embeddings' not in st.session_state:
    st.session_state['selected_embeddings'] = EMBEDDING_OPTIONS[0]
if 'pdf_file' not in st.session_state:
    st.session_state['pdf_file'] = None
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None

##################
    
# Title
col1, col2 = st.columns([5, 20])
with col2:
    st.title(TITLE)

with st.sidebar:
    tabs = st.sidebar.selectbox('SELECT TASK', [
        "Question & Answer",
        "Transcript Intelligence",
    ], on_change=reset_session_state)
    st.markdown('---')
    # Filters
    if tabs=='Question & Answer':
        selected_model = st.selectbox("Select Model:", options=MODEL_OPTIONS, index=2)
        selected_embeddings = st.selectbox("Select Embeddings:", options=EMBEDDING_OPTIONS, index=0)
        strategy = st.selectbox("Select Strategy:", options=STRATEGY_OPTIONS, index=1, disabled=True)
    elif tabs == 'Transcript Intelligence':
        selected_model = st.selectbox("Select Model:", options=MODEL_OPTIONS, index=2)
#        selected_embeddings = st.selectbox("Select Embeddings:", options=EMBEDDING_OPTIONS, index=0)
#        claimnumber = st.selectbox("Select Claim:", options=CLAIM_OPTIONS, index=0, disabled=True)

st.error("Disclaimer: All data processed in this application will be sent to OpenAI API based in the United States.")

st.markdown('## '+tabs)

if tabs=='Question & Answer':
    pdf_file = st.file_uploader('Upload a PDF file', type=['pdf'])
    if pdf_file:
        if pdf_file!=st.session_state['pdf_file']:
            st.session_state['pdf_file'] = pdf_file
            with open(pdf_file.name, "wb") as f:
                f.write(pdf_file.getbuffer())
            with st.spinner('Creating embeddings...'):
                texts = load_pdf_document(file_path=pdf_file.name)
                retriever = get_retriever(texts, embeddings[selected_embeddings])
                st.session_state['retriever'] = retriever
            
    st.markdown('---')
    
    # Question & Answer
    if st.session_state['pdf_file'] is None:
        disable_query = True
    else:
        disable_query = False
    query = st.text_input('Input your question', disabled=disable_query)
    if strategy=='Without Chain-of-Thought':
        instruction = st.text_area('Input your instruction (optional)', disabled=disable_query)
    button_query = st.button('Submit', disabled=(query==''))
    
    if button_query:

        print('---- run query ----')
        print(f'model: {selected_model}  embeddings: {selected_embeddings}')
        if selected_embeddings!=st.session_state['selected_embeddings']:
            st.session_state['selected_embeddings'] = selected_embeddings
            texts = load_pdf_document(file_path=st.session_state['pdf_file'].name)
            st.session_state['retriever'] = get_retriever(texts, embeddings[selected_embeddings])
#        qa = RetrievalQA.from_chain_type(llm=models[selected_model], chain_type="stuff", 
#                                         retriever=st.session_state['retriever'], return_source_documents=True)
        docs = st.session_state['retriever'].get_relevant_documents(query)
        context = '\n\n'.join([doc.page_content for doc in docs])
        
        if strategy=='Without Chain-of-Thought':
            user_token = model_configs[selected_model]['USER_TOKEN']
            end_token = model_configs[selected_model]['END_TOKEN']
            assistant_token = model_configs[selected_model]['ASSISTANT_TOKEN']
    #        prompt = user_token + query + end_token + assistant_token
    #        answer = qa({"query": prompt})
    #        st.write(answer['result'])
    
    #        prompt_pattern = user_token + "Here is the context:\n\n" + "{context}" + \
    #                '\n\nPlease answer below question.\n\n' + query + end_token + assistant_token
            prompt_pattern = f"{user_token}{instruction}\n\nContext: {{context}}\n\nQuestion: {query}\n\nAnswer:{end_token}{assistant_token}"
            prompt = prompt_pattern.replace('{context}', context)
    #        print(prompt)
            updated_context = truncate_context(prompt_pattern, context, 
                                               max_token_len=model_configs[selected_model]['MAX_TOKENS'],
                                               max_new_token_length=model_configs[selected_model]['MAX_NEW_TOKEN_LENGTH'])
            updated_prompt = prompt_pattern.replace('{context}', updated_context)
            with st.spinner():
                answer = models[selected_model].generate([updated_prompt]).generations[0][0].text.strip()
            st.write(answer)
            if updated_prompt!=prompt:
                st.caption(f"Note: The context has been truncated to fit model max tokens of {model_configs[selected_model]['MAX_TOKENS']}. Original context contains {len(context.split())} words. Truncated context contains {len(updated_context.split())} words.")

        else:
            chain = PDSCoverageChain()
            with st.spinner():
                answer = chain.generate(models[selected_model], model_configs[selected_model], query, context)
            st.write(answer)
           
        with st.expander("References"):
            for doc in docs:
                st.markdown('###### Page {}'.format(doc.metadata['page']))
                st.write(doc.page_content.replace('\n','\n\n'))
        
elif tabs == 'Transcript Intelligence':
    transcript_json = st.selectbox("Select sample transcript:", 
                                   [
                                       '',
                                       'sample_transcript1',
                                       'sample_transcript2'
                                    ])
    PROMPT_TEMPLATE_OPTIONS = [
"Here is a call transcript between a customer and an agent. Summarise it to understand the topic of the call, customer and agent's sentiments, the call outcome, and next steps.",
'Create custom prompt template',
    ]
    prompt_template = st.radio("Select prompt template:", PROMPT_TEMPLATE_OPTIONS)
    if prompt_template=='Create custom prompt template':
        prompt_template = st.text_area('', '')
        
    st.markdown('---')
    
    if (transcript_json!='') and (prompt_template!=''):
        TRANSCRIPT_PATH = 'sample_data/transcript/'        
        with open(TRANSCRIPT_PATH+transcript_json+'.json', 'r') as f:
            transcript = json.load(f)
        transcript = convert_transcript(transcript)
            
        user_token, end_token, assistant_token = get_special_tokens(selected_model)
        
        prompt_pattern = user_token + prompt_template + '\n' + '{context}' + end_token + assistant_token
        prompt = prompt_pattern.replace('{context}', transcript)
        updated_context = truncate_context(prompt_pattern, context=transcript, 
                                           max_token_len=model_configs[selected_model]['MAX_TOKENS'],
                                            max_new_token_length=model_configs[selected_model]['MAX_NEW_TOKEN_LENGTH'])
        updated_prompt = prompt_pattern.replace('{context}', updated_context)

        with st.spinner('Generating summary...'):
            if selected_model=='TheBloke/stable-vicuna-13B-HF':
                transcript_summary = model_pipeline.get_inference(models[selected_model], updated_prompt)
            else:
                transcript_summary = models[selected_model].generate([updated_prompt]).generations[0][0].text.strip()
        
        st.markdown("#### Transcript")
        with st.expander("Original transcript"):
            st.markdown(transcript)
        st.markdown("#### Transcript Summary")
        st.write(transcript_summary)
        if updated_prompt!=prompt:
            st.caption(f"Note: The context has been truncated to fit model max tokens of {model_configs[selected_model]['MAX_TOKENS']}. Original context contains {len(transcript.split())} words. Truncated context contains {len(updated_context.split())} words.")
