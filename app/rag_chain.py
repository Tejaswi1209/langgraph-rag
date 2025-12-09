import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
 
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
API_KEY = os.getenv("OPENAI_API_KEY")
 
llm = AzureChatOpenAI(
    openai_api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name=AZURE_DEPLOYMENT,
    api_version=API_VERSION,
    temperature=0.2,
)
 
embeddings = AzureOpenAIEmbeddings(
    openai_api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    deployment=AZURE_EMBEDDING_DEPLOYMENT,
    chunk_size=1000,
)
 
def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        input_key="query",   
        output_key="result" 
    )