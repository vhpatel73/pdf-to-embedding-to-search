# Q&A with PDF document(s)

This project implements the **Document Embedding Search** pattern for searching PDF documents. The technique uses a combination of embedding and LLM. Embedding is a process of converting text into a vector representation that captures the meaning of the text. LLM is a large language model that can be used to understand the meaning of text.

The steps followed to perform Document Embedding Search are:

1. Extract text from PDF document(s) - This step is implemented using langchain's document loader and PyPDF libraries.

2. Split documents into text chunks - This next step is to split documents into manageable text chunks. It is accomplished by using langchain's `RecursiveCharacterTextSplitter`. 

3. Create document embeddings - The third step is to create embeddings for each chunk. Few options are implemented for these steps. The first option is to use GCP Vertex AI embedding. The second option implemented here uses the huggingface embedding library to allow the number of opensource embedding models (i.e. all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-mpnet-base-dot-v1, etc)

4. Save and index embeddings - This step uses a couple of open-source Vector Stores to save and index embeddings. Chroma and FAISS are supported in this implementation.

5. Search the embeddings - This step is to search the embeddings for the search query and use LLM to craft the result. GCP Vertex AI PaLM (text-bison) is used for this step. 

Overall, **Document Embedding Search** is a powerful technique for searching PDF documents to find relevant information to search query even if the query does not contain any keywords that are found in the document. This approach is scalable but requires the right combination of embedding, similarity search technique, and a large language model. The goal of this project is to make it easy to try various embedding models and vector stores for comparative study. The choice of embedding model and vector stores are externalized in [app.cfg](app.cfg) configuration file to make it easy to try multiple permutations.  

## Supported Stack

* Supported Text Embedding Models
    - GCP Vertex AI PaLM (textembedding-gecko)
    - sentence-transformers/all-MiniLM-L6-v2
    - sentence-transformers/all-mpnet-base-v2
    - sentence-transformers/multi-qa-mpnet-base-dot-v1
* Vector Store
    - Chroma
    - FAISS
* LLM
    - GCP Vertex AI PaLM (text-bison)

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
- Configure appropriate values as desire in `app.cfg`
- To run : `python app.py`

