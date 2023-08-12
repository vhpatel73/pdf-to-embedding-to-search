# Q&A with PDF document(s)

This project converts a set of PDFs into text chunks, converts them into various embedding models, stores and indexes them with various vector databases, and leverages Vertex LLM to power semantic search. This project is useful for anyone who wants to create a semantic search engine for PDF documents.

The primary goal of this project is to make it easy to try various embedding models and vector databases for comparative study. Use [app.cfg](app.cfg) configuration file to switch embedding models, vector database, and llm. 


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

### Configure & Project 

- Copy PDF files to be processed in `input` folder
- Configure appropriate values as desire in `app.cfg`
- To run : `python app.py`

