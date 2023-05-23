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

#from langchain.chains import RetrievalQA

#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import json
import pandas as pd

import model as model_pipeline

st.set_page_config(layout="wide")

# Parameters
TITLE = 'SunGPT Demo App'
MODEL_OPTIONS = [
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

CLAIM_OPTIONS = [
    'H12345678',
    'H87654321',
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
    embeddings.update({
        'OpenAIEmbeddings': OpenAIEmbeddings()
    })
#    embeddings.update({
#        'InstructEmbeddings-Large': HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
#    })
#    embeddings.update({
#        'InstructEmbeddings-XL': HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
#    })
    
    return embeddings

@st.cache_resource
def init_models():
    print('Initializing models....')
    models = {}
    # OpenAI
    models.update({
        'OpenAI/GPT-3.5-turbo': OpenAI(
            model_name='gpt-3.5-turbo',
            max_tokens=256,
            top_p=0.95,
                                                                        temperature=0.001,
        )
    })
    # Pythia
#    MODEL_NAME='OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'
#    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map='auto', load_in_8bit=True)
#    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#    hf_pipeline = pipeline('text2text-generation', model=hf_model, tokenizer=hf_tokenizer, length=2048, temperature=0)
#    models.update({
#        'OpenAssistant/SFT-4 Pythia 12B': HuggingFacePipeline(pipeline=hf_pipeline)
#    })
    MODEL_NAME = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'
    client = InferenceAPIClient(MODEL_NAME, token=os.getenv("HF_TOKEN", None))
    model = HuggingFaceTextGenInference(
        inference_server_url='https://api-inference.huggingface.co/models/'+MODEL_NAME,
        max_new_tokens=256,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.001,
        repetition_penalty=1.03,
        client=client,
    )
    models.update({
        'OpenAssistant/SFT-4 Pythia 12B': model
    }),
    
#    MODEL_NAME = 'TheBloke/stable-vicuna-13B-HF'
#    model = HuggingFaceHub(repo_id=MODEL_NAME, model_kwargs={"temperature":0})
#    model = model_pipeline.get_model_pipeline(MODEL_NAME)
#    models.update({
#        'TheBloke/stable-vicuna-13B-HF': model
#    }),
    
    return models

@st.cache_resource
def init_app():
    texts = None
    embeddings = init_embeddings()
    models = init_models()
    
    return texts, models, embeddings

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
    text = '\n\n'.join(pd.DataFrame(transcript)[['sp','w']].apply(lambda x: x.sp+': '+x.w, axis='columns'))
    return text

def reset_session_state():
    st.session_state['selected_embeddings'] = EMBEDDING_OPTIONS[0]
    st.session_state['pdf_file'] = None
    st.session_state['retriever'] = None
    
# Model initialization    
texts, models, embeddings = init_app()

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
        selected_model = st.selectbox("Select Model:", options=MODEL_OPTIONS, index=0)
        selected_embeddings = st.selectbox("Select Embeddings:", options=EMBEDDING_OPTIONS, index=0)
        strategy = st.selectbox("Select Strategy:", options=STRATEGY_OPTIONS, index=1, disabled=True)
    elif tabs == 'Transcript Intelligence':
        selected_model = st.selectbox("Select Model:", options=MODEL_OPTIONS, index=0)
#        selected_embeddings = st.selectbox("Select Embeddings:", options=EMBEDDING_OPTIONS, index=0)
#        claimnumber = st.selectbox("Select Claim:", options=CLAIM_OPTIONS, index=0, disabled=True)

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
    query = st.text_input('Input your question here and press Enter', disabled=disable_query)
    
    if query:
        print('---- run query ----')
        print(f'selected_embeddings: {selected_embeddings}')
        print(f"st.session_state['selected_embeddings']: {st.session_state['selected_embeddings']}")
        if selected_embeddings!=st.session_state['selected_embeddings']:
            st.session_state['selected_embeddings'] = selected_embeddings
            texts = load_pdf_document(file_path=st.session_state['pdf_file'].name)
            st.session_state['retriever'] = get_retriever(texts, embeddings[selected_embeddings])
#        qa = RetrievalQA.from_chain_type(llm=models[selected_model], chain_type="stuff", 
#                                         retriever=st.session_state['retriever'], return_source_documents=True)
        docs = st.session_state['retriever'].get_relevant_documents(query)
        context = '\n\n'.join([doc.page_content for doc in docs])
        
        user_token, end_token, assistant_token = get_special_tokens(selected_model)
#        prompt = user_token + query + end_token + assistant_token
#        answer = qa({"query": prompt})
#        st.write(answer['result'])
        prompt = user_token + "Here is the context:\n\n" + context + \
                '\n\nPlease answer below question.\n\n' + query + end_token + assistant_token
        answer = models[selected_model].generate([prompt]).generations[0][0].text.strip()
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
"""Here is a call transcript between our customer with our consutlant. 
Summarize it to understand the topic of the call, customer sentiment, and consultant's empathy.
""",
"""Summarize below call transcript between the agent and a customer. 
The summary should contain the reason for customer's call, agent's solution and outcome. 
Be factual. 
Let us think step by step to make sure the summary is correct.
""",
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
#        print(transcript)
            
        user_token, end_token, assistant_token = get_special_tokens(selected_model)
        prompt = user_token + prompt_template + '\n' + transcript + end_token + assistant_token
        if selected_model=='TheBloke/stable-vicuna-13B-HF':
            transcript_summary = model_pipeline.get_inference(models[selected_model], prompt)
        else:
            transcript_summary = models[selected_model].generate([prompt]).generations[0][0].text.strip()
#        print(transcript_summary)
        
        st.markdown("#### Transcript")
        with st.expander("Original transcript"):
            st.markdown(transcript)
        st.markdown("#### Transcript Summary")
        st.write(transcript_summary)
    
        st.markdown("#### Category Hit")
        category_hit = """\
        Hello World"""
        st.write(category_hit)
        
