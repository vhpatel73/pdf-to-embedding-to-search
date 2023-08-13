import chainlit as cl
from chainlit.input_widget import Select
import app 

def init_chain(embeddding, vectordb, llm):
    app.EMBEDDING = embeddding
    app.VECTOR_DB = vectordb
    app.LLM = llm
    app.USECACHE = 'true'
    app.VECTOR_DB_LOC = f'{app.DB_PATH}/{app.VECTOR_DB}/{app.EMBEDDING}'
    embeddings = app.get_embedding_model()
    retriever = app.get_retriever(embeddings)
    return app.retrievalQA(app.get_llm_model(), "stuff", retriever)

@cl.on_chat_start
async def start():
    cl.user_session.set("llm_chain", init_chain("sentence-transformers/all-MiniLM-L6-v2","FAISS","vertexllm"))
    settings = await cl.ChatSettings(
        [
            Select(
                id="Embedding",
                label="Embedding",
                values=["sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/all-mpnet-base-v2",
                        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                        "textembedding-gecko"],
                initial_index=0,
            ),
            Select(
                id="Vector_Store",
                label="Vector Store",
                values=["FAISS","Chroma"],
                initial_index=0,        
            ),
            Select(
                id="LLM",
                label="LLM",
                values=["vertexllm"],
                initial_index=0,        
            ),
        ]
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings["Embedding"], settings["Vector_Store"])    
    cl.user_session.set("llm_chain", init_chain(settings["Embedding"],settings["Vector_Store"],settings["LLM"]))

@cl.on_message
async def main(message: str):
    retrievalQA = cl.user_session.get("llm_chain")
    res = await retrievalQA.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["result"]).send()