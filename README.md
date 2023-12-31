# Q&A with PDF document(s)

This project implements the **Retrieval Augmented Generation** pattern for searching proprietary knowledgebase. The technique uses a combination of embedding technique and pre-trained LLM. Embedding is a process of converting text into a vector representation that captures the meaning of the text. LLM is a large language model that can be used to understand the meaning of text.

The steps followed to perform RAG are:

1. **Extract text from PDF document(s)** - This step is implemented using langchain's document loader and PyPDF libraries.

2. **Split documents into text chunks** - This next step is to split documents into manageable text chunks. It is accomplished by using langchain's `RecursiveCharacterTextSplitter`. 

3. **Create document embeddings** - The third step is to create embeddings for each chunk. Few options are implemented for these steps. The first option is to use GCP Vertex AI embedding. The second option implemented here uses the huggingface embedding library to allow the number of opensource embedding models (i.e. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1), etc)

4. **Save and index embeddings** - This step uses a couple of open-source Vector Stores to save and index embeddings. [Chroma](https://github.com/chroma-core/chroma) and [FAISS](https://ai.meta.com/tools/faiss/) are supported in this implementation.

5. **Search the embeddings** - This step is to search the embeddings for the search query and use LLM to craft the result. [GCP Vertex AI PaLM](https://cloud.google.com/blog/products/ai-machine-learning/generative-ai-applications-with-vertex-ai-palm-2-models-and-langchain) is used for this step. 

Overall, **Retrieval Augmented Generation** is a powerful technique for searching proprietary knowledgebase to find relevant information to search query even if the query does not contain any keywords that are found in the document. This approach is scalable but requires the right combination of embedding, similarity search technique, and a large language model. The goal of this project is to make it easy to try various embedding models and vector stores for comparative study. The choice of embedding model and vector stores are externalized in [app.cfg](app.cfg) configuration file to make it easy to try multiple permutations.  

## Supported Stack

* Supported Text Embedding Models
    - [GCP Vertex AI PaLM (textembedding-gecko)](https://cloud.google.com/blog/products/ai-machine-learning/generative-ai-applications-with-vertex-ai-palm-2-models-and-langchain) 
    - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    - [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
    - [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
* Vector Store
    - [Chroma](https://github.com/chroma-core/chroma)
    - [FAISS](https://ai.meta.com/tools/faiss/)
* LLM
    - [GCP Vertex AI PaLM (text-bison)](https://cloud.google.com/blog/products/ai-machine-learning/generative-ai-applications-with-vertex-ai-palm-2-models-and-langchain)

### Setup Project Workspace

Clone repo on your workspace
```
git clone https://github.com/vhpatel73/pdf-to-embedding-to-search.git
```

Create virtual environment
```
python -m venv venv
```

Install necessary packages
```
source venv/bin/activate

pip install -r requirements.txt
```

Create input and output folders
```
mkdir input
mkdir output
```

To use Vertex AI PaLM you must have the google-cloud-aiplatform Python package installed and either:

* Have credentials configured for your environment (gcloud, workload identity, etc...)
* Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable

For detail information, see: [Setup GCP](https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth) & [Google auth](https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth)

### Configuration 

- Copy PDF files to be processed in `input` folder
- Configure appropriate values as desire in `app.cfg`. It is recommended to run this program with different combination of embeddding models and vector stores in `app.cfg` while keeping `reuse_index = false`. 
- To run : `python app.py`

### Test

* After generating vector stores,
    - Configure `questions` and `testcases` in `test/bulktest.py`.
    - Run `python test/bulktest.py`
    - Record and score the results

### UI - Portal

* UI support is added using `Chainlit`. 
* In `chatbot.py`, configure `settings` variable with supported models available in your vector stores.
* To run - `chainlit run chatbot.py`

## References

* Retrieval Augmented Generation : [Ref1](https://arxiv.org/abs/2005.11401) - [Ref2](https://huggingface.co/docs/transformers/model_doc/rag)
* GCP Vertex AI PaLM (textembedding-gecko) : [Ref](https://cloud.google.com/blog/products/ai-machine-learning/generative-ai-applications-with-vertex-ai-palm-2-models-and-langchain) 
* sentence-transformers/all-MiniLM-L6-v2 : [Ref](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* sentence-transformers/all-mpnet-base-v2 : [Ref](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
* sentence-transformers/multi-qa-mpnet-base-dot-v1 : [Ref](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
* Chroma : [Ref](https://github.com/chroma-core/chroma)
* FAISS : [Ref](https://ai.meta.com/tools/faiss/)

