
# LangGraph-based graph definition for RAG workflow
from langgraph.graph import StateGraph, END
from app.loader import extract_text
from app.splitter import split_text
from app.embedder import build_vector_store
from app.rag_chain import create_rag_chain

def entry_node(state):
    # Entry node: expects file and filename in state
    return state

def extract_text_node(state):
    file_path = state.get("file")
    filename = state.get("filename")
    if not file_path:
        return {**state, "answer": "❌ No file provided."}
    text = extract_text(file_path, filename)
    if not text.strip():
        return {**state, "answer": "❌ No content extracted from file."}
    return {**state, "raw_text": text}

def split_text_node(state):
    text = state.get("raw_text", "")
    if not text.strip():
        return {**state, "answer": "❌ No content to split."}
    chunks = split_text(text)
    if not chunks:
        return {**state, "answer": "❌ No content to split."}
    return {**state, "chunks": chunks}

def build_vector_store_node(state):
    chunks = state.get("chunks", [])
    if not chunks:
        return {**state, "answer": "❌ No chunks to embed."}
    vector_store = build_vector_store(chunks)
    return {**state, "vector_store": vector_store}

def answer_node(state):
    query = state.get("query")
    vector_store = state.get("vector_store")
    if not query:
        return {**state, "answer": "❌ No query provided."}
    if not vector_store:
        return {**state, "answer": "❌ Missing vector store for answering."}
    try:
        qa_chain = create_rag_chain(vector_store)
        response = qa_chain.invoke({"query": query})
        return {**state, "answer": response.get("result", "⚠️ No answer generated.")}
    except Exception as e:
        return {**state, "answer": f"❌ Error generating answer: {str(e)}"}

# Build the LangGraph graph
def build_rag_graph():
    graph = StateGraph()
    # Add nodes
    graph.add_node("entry", entry_node)
    graph.add_node("extract_text", extract_text_node)
    graph.add_node("split_text", split_text_node)
    graph.add_node("build_vector_store", build_vector_store_node)
    graph.add_node("answer", answer_node)
    # Add edges
    graph.add_edge("entry", "extract_text")
    graph.add_edge("extract_text", "split_text")
    graph.add_edge("split_text", "build_vector_store")
    graph.add_edge("build_vector_store", "answer")
    graph.add_edge("answer", END)
    # Set entry and exit
    graph.set_entry_point("entry")
    graph.set_exit_point(END)
    return graph