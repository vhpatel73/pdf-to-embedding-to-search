'''

Run set of questions against multiple embedding/llm models.
This test program assumes that vector store for each combinations listed in 'test_config' are created.

'''
import app
import time

questions = [
    "What services AAA provides?",
    "Where is AAA headquaters located",
    "When US stamps issued to promote the school safety patrol?",
    "What are the initiates AAA took on by engaging with US Goverment?",
    "What is Sportsmanlike driving?",
    "What is Trip Tik?",
    "Name few regulatory acts where AAA played significant role."
]

test_config = [
    ['textembedding-gecko','Chroma','vertexllm'],
    ['textembedding-gecko','FAISS','vertexllm'],
    ['sentence-transformers/all-MiniLM-L6-v2','Chroma','vertexllm'],
    ['sentence-transformers/all-MiniLM-L6-v2','FAISS','vertexllm'],
    ['sentence-transformers/all-mpnet-base-v2','Chroma','vertexllm'],
]
resultsets = []

for testno, testcase in enumerate(test_config):
    app.EMBEDDING = testcase[0]
    app.VECTOR_DB = testcase[1]
    app.LLM = testcase[2]
    app.VECTOR_DB_LOC = f'{app.DB_PATH}/{app.VECTOR_DB}/{app.EMBEDDING}'
    app.USECACHE = 'true'

    embeddings = app.get_embedding_model()
    retriever = app.get_retriever(embeddings)
    retrievalQA = app.retrievalQA(app.get_llm_model(), "stuff", retriever)
    print(f'\n*** *** ***\n')
    print(f'[{testno}] : [{app.EMBEDDING}] - [{app.VECTOR_DB}] - [{app.LLM}]')
    print(f'\n*** *** ***\n')
    resultset = []
    for q in questions:
        time.sleep(5)
        result = app.getAnswer(retrievalQA, q)
        resultset.append(result["result"])
    resultsets.append(resultset)

total_testcases = len(test_config)
total_questions = len(questions)

# Display Results
for i in range(total_questions):
    print(f'\n(Q) \t: {questions[i]}')
    for j in range(total_testcases):
        print(f'(A-{j}) \t: {resultsets[j][i]}')
    