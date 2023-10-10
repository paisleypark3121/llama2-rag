import os
from dotenv import load_dotenv

import transformers
from torch import (
    bfloat16,
    cuda
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import (
    HuggingFacePipeline,
    CTransformers
)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain
)

load_dotenv()

def from_cache():

    #Note that running this on CPU is sloooow
    # If running on Google Colab you can avoid this by going to Runtime
    # Change runtime type > Hardware accelerator > GPU > GPU type > T4
    # This should be included within the free tier of Colab
    # it will be necessary to let the LLAMA2 model be downloaded on colab
    # https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/llama-2/llama-2-13b-retrievalqa.ipynb

    model_id="meta-llama/Llama-2-7b-chat-hf"
    HF_AUTH_TOKEN=os.getenv('HF_AUTH_TOKEN')

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=HF_AUTH_TOKEN
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=HF_AUTH_TOKEN
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=HF_AUTH_TOKEN
    )

    file_path="data\PT707-Transcript.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunk_size=1200
    chunk_overlap=200

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        length_function=len
    )

    splits = r_splitter.split_documents(docs)

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model_id="sentence-transformers/all-MiniLM-L6-v2"
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    persist_directory="chroma"
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embed_model,
        persist_directory=persist_directory
    )

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  
        task='text-generation',
        temperature=0.2,  
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=vectordb.as_retriever()
    )

    query="What is Grafbase?"
    response=rag_pipeline(query)
    print(response)

    # response:
    #{'query': 'What is Grafbase?', 'result': ' Grafbase is a tool that allows you to unify, extend, and cache your data sources via a single GraphQ L API deployed to the edge closest to your web and mobile users. It also makes it effortless to turn OpenAPI or MongoDB sources into GraphQL APIs, and provides a command-line interface for building and deploying APIs locally, with automatic creation of preview deployment APIs for easy testing and collaboration.'}

def from_folder():

    file_path="data\PT707-Transcript.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunk_size=500
    chunk_overlap=50

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        length_function=len
    )

    splits = r_splitter.split_documents(docs)

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model_id="sentence-transformers/all-MiniLM-L6-v2"
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    persist_directory="chroma"
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embed_model,
        persist_directory=persist_directory
    )

    llm = CTransformers(
        model="../llama2/llama-2-13b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens':128,'temperature':0.01})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={"k":2}),
        memory=memory)

    query="What is Grafbase?"
    response = chain.run(query)
    print(response)

    # response:
    #{'query': 'What is Grafbase?', 'result': ' Grafbase is a tool that allows you to unify, extend, and cache your data sources via a single GraphQ L API deployed to the edge closest to your web and mobile users. It also makes it effortless to turn OpenAPI or MongoDB sources into GraphQL APIs, and provides a command-line interface for building and deploying APIs locally, with automatic creation of preview deployment APIs for easy testing and collaboration.'}    

#from_cache()
from_folder()