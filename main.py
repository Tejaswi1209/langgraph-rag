import chainlit as cl
from app.graph import build_rag_graph
 
# 🔹 Shared state

STATE = {
    "file": None,
    "filename": None,
    "query": None,
    "raw_text": None,
    "chunks": None,
    "vector_store": None,
    "answer": None,
}
 
@cl.on_chat_start
async def start():
    await cl.Message("📎 Upload a file (txt, docx, pdf) to start.").send()
 
@cl.on_message
async def handle_message(message: cl.Message):
    global STATE
    content = message.content.strip()
    upload = message.elements[0] if message.elements else None
    graph = build_rag_graph()
    if upload:
        # Reset STATE
        STATE.update({
            "file": upload.path,
            "filename": upload.name,
            "query": None,
            "raw_text": None,
            "chunks": None,
            "vector_store": None,
            "answer": None
        })
        await cl.Message(f"✅ File uploaded successfully: {upload.name}").send()
        # Run graph up to vector store creation
        # (entry -> extract_text -> split_text -> build_vector_store)
        state = graph.run_until(
            STATE,
            stop_at_nodes=["answer"]
        )
        STATE.update(state)
        if STATE.get("answer") and "❌" in STATE["answer"]:
            await cl.Message(STATE["answer"]).send()
            return
        await cl.Message("💬 File processed. Ask a question!").send()
    elif content:
        STATE["query"] = content
        # Only run answer node
        state = graph.run_until(
            STATE,
            stop_at_nodes=["answer"]
        )
        STATE.update(state)
        await cl.Message(STATE.get("answer", "⚠️ No answer generated")).send()
    else:
        await cl.Message("❌ Please upload a file or ask a valid question.").send()

 