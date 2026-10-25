import shutil
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma4:31b-cloud"
CHROMA_DIR = "./chroma_db"

st.set_page_config(page_title="Lecture Chat", page_icon="📚", layout="wide")


# ---------- models (loaded once) ----------
@st.cache_resource
def load_models():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return emb, llm, reranker


emb, llm, reranker = load_models()

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using only the context below. "
    "If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


# ---------- indexing ----------
def build_index(folder, chunk_size, overlap):
    files = sorted(Path(folder).rglob("*.pdf"))  # rglob = includes subfolders
    if not files:
        return None, 0, 0
    docs = []
    for f in files:
        try:
            docs.extend(PyPDFLoader(str(f)).load())
        except Exception as e:
            st.warning(f"Skipped {f.name}: {e}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    store = Chroma.from_documents(chunks, emb, persist_directory=CHROMA_DIR)
    return store, len(files), len(chunks)


def get_store():
    if "store" not in st.session_state and Path(CHROMA_DIR).exists():
        st.session_state.store = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
    return st.session_state.get("store")


# ---------- RAG steps ----------
def expand(q):
    out = llm.invoke(
        "Fix the spelling of this question, then write 2 alternative phrasings "
        "using synonyms. Return one per line, nothing else.\n\nQuestion: " + q
    ).content
    lines = [l.strip() for l in out.split("\n") if l.strip()][:3]
    return lines or [q]


def rerank(question, chunks, top_n):
    scores = reranker.predict([(question, c.page_content) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return ranked[:top_n]


def ask(question, store, top_k, history):
    search_q = question
    if history:
        past = "\n".join(f"User: {q}\nAI: {a}" for q, a in history[-3:])
        search_q = llm.invoke(
            f"Chat history:\n{past}\n\nRewrite this new question so it makes sense "
            f"on its own. Return only the question.\n\nNew question: {question}"
        ).content

    variants = expand(search_q)
    candidates = {}
    for v in variants:
        for d in store.similarity_search(v, k=6):
            candidates[d.page_content] = d
    top = rerank(variants[0], list(candidates.values()), top_k)

    context = "\n\n".join(c.page_content for _, c in top)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": variants[0]}
    )
    sources = [
        (Path(c.metadata["source"]).name, c.metadata["page"] + 1, float(s), c.page_content)
        for s, c in top
    ]
    return answer, variants, sources


# ---------- sidebar ----------
with st.sidebar:
    st.header("📁 Documents")
    folder = st.text_input("Folder with PDFs", value="./lectures")
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
    overlap = st.slider("Chunk overlap", 0, 400, 150, 50)
    if st.button("Build / rebuild index", type="primary"):
        if not Path(folder).is_dir():
            st.error("Folder not found.")
        else:
            with st.spinner("Reading and embedding PDFs..."):
                store, n_files, n_chunks = build_index(folder, chunk_size, overlap)
            if store is None:
                st.error("No PDFs found in that folder.")
            else:
                st.session_state.store = store
                st.success(f"Indexed {n_files} PDFs, {n_chunks} chunks.")

    st.header("⚙️ Settings")
    top_k = st.slider("Chunks sent to the model", 1, 10, 4)
    show_debug = st.checkbox("Show search variants and scores", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ---------- chat ----------
st.title("📚 Chat with your lectures")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])
store = get_store()

if store is None:
    st.info("Put your PDFs in a folder, enter its path in the sidebar, and click **Build / rebuild index**.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for name, page, score, text in m["sources"]:
                    st.markdown(f"**{name}**, page {page}" + (f" (score {score:.2f})" if show_debug else ""))
                    st.caption(text[:300] + "...")

question = st.chat_input("Ask about your lectures", disabled=store is None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, variants, sources = ask(question, store, top_k, st.session_state.history)
        st.markdown(answer)
        if show_debug:
            st.caption("Searched for: " + " | ".join(variants))
        with st.expander("Sources"):
            for name, page, score, text in sources:
                st.markdown(f"**{name}**, page {page}" + (f" (score {score:.2f})" if show_debug else ""))
                st.caption(text[:300] + "...")

    st.session_state.history.append((question, answer))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
