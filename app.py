from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain import PromptTemplate
import configparser

config = configparser.ConfigParser()
config.read("app.cfg")

DATA_PATH = str(config["data"]["input"])
DB_PATH = str(config["data"]["db"])
CHUNK_SIZE = int(config["data"]["chunksize"])
CHUNK_OVERLAP = int(config["data"]["overlap"])
EMBEDDING = str(config["embedding"]["model"])
VECTOR_DB = str(config["vector"]["db"])
VECTOR_DB_LOC = f'{DB_PATH}/{VECTOR_DB}/{EMBEDDING}'
USECACHE = config["vector"]["reuse_index"]
LLM = str(config["llm"]["model"])
PROMPT_TEMPLATE = str(config["prompt"]["template"])


def create_text_chunks():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    log(f'{len(documents)} docs will be processed from {DATA_PATH}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    log(f'{len(texts)} Text chunks to be converted into embedding')
    return texts


def get_HuggingFace_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING, model_kwargs={'device': 'cpu'})
    log(f'Setting "{EMBEDDING}" as embedding model')
    return embeddings


def get_vertex_embedding_model():
    embeddings = VertexAIEmbeddings(requests_per_minute=150)
    log(f'Setting "Vertex Embedding (textembedding-gecko)" as embedding model')
    return embeddings


def get_embedding_model():
    if EMBEDDING == "textembedding-gecko":
        return get_vertex_embedding_model()
    else:
        return get_HuggingFace_embedding_model()


def create_faiss_db(texts, embeddings):
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_DB_LOC)
    log(f'Setting "FAISS" as vector database @ [{VECTOR_DB_LOC}]')


def create_chroma_db(texts, embeddings):
    db = Chroma(collection_name="langchain", persist_directory=VECTOR_DB_LOC, embedding_function=embeddings)
    db.add_documents(documents=texts, embedding=embeddings)
    db.persist()  
    log(f'Setting "Chroma" as vector database @ [{VECTOR_DB_LOC}]')


def create_db(texts, embeddings):
    if VECTOR_DB == "Chroma":
        create_chroma_db(texts, embeddings)
    else:
        create_faiss_db(texts, embeddings)


def get_chroma_retriever(embeddings):
    db = Chroma(collection_name="langchain", persist_directory=VECTOR_DB_LOC, embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k":2})
    log(f'Chroma index loaded for Q&A')
    return retriever


def get_faiss_retriever(embeddings):
    db = FAISS.load_local(VECTOR_DB_LOC, embeddings)
    retriever = db.as_retriever(
        search_kwargs={'k': 2})
    log(f'FAISS index loaded for Q&A')
    return retriever


def get_retriever(embeddings):
    if VECTOR_DB == "FAISS":
        return get_faiss_retriever(embeddings)
    else:
        return get_chroma_retriever(embeddings)


def get_llm_model():
    log(f'Setting Google "text-bison@001" as Large Language Model')
    return VertexAI(model_name = 'text-bison@001', max_output_tokens = 256, temperature = 0.1, top_p = 0.8, top_k = 40, verbose = True,)


def set_custom_prompt():
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=['context', 'question'])
    log(f'Custom prompt created')
    return prompt


def retrievalQA(llm,chain_type,retriever):
    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()})
    log(f'Initialized Q&A chain using "LangChain"')
    return retrievalQA


def getAnswer(retrievalQA, question):
    log(f'Processing prompt ... ... ... \n')
    return retrievalQA({"query": question})    


def log(line):
    print(f' - {line}')


if __name__ == "__main__":
    print(f' * DATA_PATH = [{DATA_PATH}]')
    print(f' * DB_PATH = [{DB_PATH}]')
    print(f' * CHUNK_SIZE = [{CHUNK_SIZE}]')
    print(f' * CHUNK_OVERLAP = [{CHUNK_OVERLAP}]')
    print(f' * EMBEDDING = [{EMBEDDING}]')
    print(f' * VECTOR_DB = [{VECTOR_DB}]')
    print(f' * LLM = [{LLM}]')
    print(f' * USECACHE = [{USECACHE}]')

    print(f'')

    embeddings = get_embedding_model()

    # Only create chunks, embedding, and vector db if USECACHE is false
    if USECACHE.upper() == "FALSE": 
        texts = create_text_chunks()
        if len(texts) == 0: 
            log('No files found.')
            exit
        else:
            create_db(texts, embeddings)
    else:
        log(f'Loading existing vector db')
    retriever = get_retriever(embeddings)
    retrievalQA = retrievalQA(get_llm_model(), "stuff", retriever)
    result = getAnswer(retrievalQA, "What AAA stands for?")
    log(f'Response: {result["result"]}\n')