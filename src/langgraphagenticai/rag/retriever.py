from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import tempfile
import os


def rag_retrieve(state: dict) -> dict:
    """
    In-memory RAG retriever using uploaded file.
    """

    uploaded_file = state.get("rag_file")
    question = state["messages"][-1].content

    if not uploaded_file:
        raise ValueError("No document uploaded for RAG")

    # 🔹 Reset pointer (important for Streamlit uploaded file)
    uploaded_file.seek(0)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load document
        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # In-memory FAISS
        db = FAISS.from_documents(chunks, embeddings)
        retrieved_docs = db.similarity_search(question, k=4)

        state["context"] = retrieved_docs
        return state

    finally:
        # 🔹 Clean temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
