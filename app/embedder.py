import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    deployment=AZURE_EMBEDDING_DEPLOYMENT,
    api_version=API_VERSION,
    chunk_size=1000
)

def build_vector_store(chunks) -> FAISS:
    """
    Build a vector store from the provided chunks of text.
    
    Args:
        chunks (list[str]): List of text chunks.
        
        
    Returns:
        FAISS: The created vector store.
    """
    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def retrieve_similar_chunks(query: str, vector_store: FAISS, k: int = 5) -> list[Document]:
    """
    Retrieve similar chunks from the vector store based on the query.
    
    Args:
        query (str): The query string to search for.
        vector_store (FAISS): The vector store to search in.
        k (int): The number of similar chunks to retrieve.
        
    Returns:
        list[Document]: List of Document objects containing similar chunks.
    """
    results = vector_store.similarity_search(query, k=k)
    return results
