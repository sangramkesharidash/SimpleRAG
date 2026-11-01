import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import glob
import re
import logging
import concurrent.futures
import streamlit as st

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Configurations ---
DATA_DIR = "./data"
CHROMA_PATH = "./chroma_db_dynamic"
LOG_FILE_PATH = os.path.join(os.getcwd(), "rag_build.log")
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # Exclude files larger than 2MB

# Setup Logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

st.set_page_config(page_title="Simple RAG", page_icon="🤖", layout="wide")
#st.title("🤖 Multi-Environment Capacity Planning RAG")
st.title("⚡ Simple RAG")
# --- File Filtering & Document Loading ---
def load_single_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    # Exclude high-frequency raw metric dumps
    if "usage" in filename.lower() and ext == ".csv":
        logging.info(f"Skipped high-frequency metric file: {filename}")
        return []

    # Exclude files exceeding size limit
    if os.path.exists(file_path) and os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
        logging.warning(f"Skipped file larger than 2MB: {filename}")
        return []

    env_match = re.search(r'/(M\d+)/', file_path)
    env_name = env_match.group(1) if env_match else "Global"

    try:
        if ext in [".yaml", ".yml", ".txt", ".log"]:
            loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
            docs = loader.load()
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
        else:
            return []

        for doc in docs:
            doc.page_content = (
                f"TARGET_ENVIRONMENT: {env_name}\n"
                f"SOURCE_FILE: {filename}\n"
                f"--- CONTENT ---\n"
                f"{doc.page_content}"
            )
            doc.metadata["environment"] = env_name
            doc.metadata["source_file"] = filename
            doc.metadata["file_path"] = file_path
        return docs
    except Exception as e:
        logging.warning(f"Skipped unreadable file '{filename}': {e}")
        return []

# --- Optimized Database Builder ---
def build_vector_db():
    logging.info("=" * 50)
    logging.info("Starting database build process...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        err_msg = f"Created empty directory at '{DATA_DIR}'."
        logging.error(err_msg)
        st.error(err_msg)
        return False

    file_paths = glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
    valid_files = [
        f for f in file_paths 
        if os.path.isfile(f) 
        and not ("usage" in os.path.basename(f).lower() and f.endswith(".csv"))
        and os.path.getsize(f) <= MAX_FILE_SIZE_BYTES
    ]
    
    if not valid_files:
        err_msg = f"No valid configuration files detected inside '{DATA_DIR}'."
        logging.warning(err_msg)
        st.error(err_msg)
        return False

    logging.info(f"Discovered {len(valid_files)} eligible files inside '{DATA_DIR}'.")

    with st.status("🚀 Processing Data & Generating Embeddings...", expanded=True) as status:
        st.write(f"📁 Processing **{len(valid_files)}** configuration & summary files...")
        
        all_docs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {executor.submit(load_single_file, path): path for path in valid_files}
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_file):
                path = future_to_file[future]
                completed_count += 1
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
                        logging.info(f"Parsed ({completed_count}/{len(valid_files)}): {path} -> {len(docs)} docs")
                except Exception as exc:
                    logging.error(f"Error reading {path}: {exc}")

                if completed_count % 250 == 0 or completed_count == len(valid_files):
                    st.write(f"✓ Parsed {completed_count}/{len(valid_files)} files...")

        if not all_docs:
            err_msg = "No document content extracted from valid files."
            logging.error(err_msg)
            st.error(err_msg)
            return False

        logging.info(f"Filtered raw documents extracted: {len(all_docs)}")
        st.write(f"✂️ Chunking {len(all_docs)} parsed documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        total_chunks = len(splits)
        
        logging.info(f"Generated {total_chunks} text chunks for indexing.")
        st.write(f"🧠 Generating vector embeddings for **{total_chunks}** chunks...")

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        batch_size = 100
        total_batches = (total_chunks - 1) // batch_size + 1
        
        for i in range(0, total_chunks, batch_size):
            batch = splits[i:i + batch_size]
            vectorstore.add_documents(batch)
            
            current_batch = (i // batch_size) + 1
            percent_complete = int((current_batch / total_batches) * 100)
            
            logging.info(f"Embedded batch {current_batch}/{total_batches} ({percent_complete}%)")
            
            progress_bar.progress(percent_complete)
            progress_text.write(f"**Embedding Progress:** {percent_complete}% (Batch {current_batch}/{total_batches})")
            
        status.update(label="✅ Indexing Complete!", state="complete", expanded=False)

    logging.info("Vector database built successfully!")
    st.success(f"Indexed {len(all_docs)} documents ({total_chunks} chunks) successfully!")
    return True

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Vector Store Controls")
    if st.button("🔨 Build / Rebuild Database"):
        if build_vector_db():
            st.cache_resource.clear()
            st.rerun()

# --- Initialize Chain ---
@st.cache_resource
def load_rag_chain():
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        return None
        
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    llm = OllamaLLM(model="mistral:7b")

    system_prompt = (
        "You are a Kubernetes infrastructure and capacity planning analyst. "
        "Use the provided context snippets to answer the question accurately.\n\n"
        "CRITICAL FORMATTING INSTRUCTIONS:\n"
        "1. Identify the 'TARGET_ENVIRONMENT' field (e.g., M1, M3, M6, M19, M21) embedded in each context snippet.\n"
        "2. When requested for CPU/Memory requests or limits across environments, generate STRICTLY a Markdown Table with these 6 exact columns:\n"
        "| Environment | Service Name | CPU Request | CPU Limit | Memory Request | Memory Limit |\n"
        "3. DO NOT output Python, shell, or HTTP code snippets unless explicitly requested.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = load_rag_chain()

if rag_chain is None:
    st.info("👈 No vector database found. Copy environment folders into `./data` and click 'Build / Rebuild Database' in the sidebar.")
    st.stop()

# --- Interactive Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 Referenced Source Files"):
                for src in message["sources"]:
                    st.write(f"- `{src}`")

if user_input := st.chat_input("Ask a question across environments..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving data across environments..."):
            res = rag_chain.invoke({"input": user_input})
            answer = res["answer"]
            
            sources = list(set(
                f"[{doc.metadata.get('environment', 'Global')}] {doc.metadata.get('source_file', 'Unknown')}" 
                for doc in res["context"]
            ))

            st.markdown(answer)
            if sources:
                with st.expander("📄 Referenced Source Files"):
                    for src in sources:
                        st.write(f"- `{src}`")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })
