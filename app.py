import streamlit as st
import fitz
import numpy as np
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(layout="wide")
st.title("📄 Multi-PDF RAG Chatbot")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_metadata" not in st.session_state:
    st.session_state.all_metadata = []

st.sidebar.header("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
user_query = st.sidebar.text_input("🔍 Ask something")

if uploaded_files and st.sidebar.button("🔄 Process PDFs"):
    with st.spinner("Indexing..."):
        all_texts = []
        all_metadata = []
        for pdf_file in uploaded_files:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                all_texts.append(text)
                all_metadata.append({"filename": pdf_file.name, "page": page_num+1})
            doc.close()
        
        st.session_state.vectorstore = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadata)
        st.session_state.all_metadata = all_metadata
        st.success(f"✅ Indexed {len(all_texts)} pages!")

if st.session_state.vectorstore and user_query:
    docs = st.session_state.vectorstore.similarity_search(user_query, k=3)
    st.subheader("🔍 Results")
    for i, doc in enumerate(docs):
        meta = st.session_state.all_metadata[i]
        st.write(f"**{meta['filename']} (Page {meta['page']})**")
        st.write(doc.page_content[:300] + "...")
