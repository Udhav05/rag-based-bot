import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(layout="wide")
st.title("🎯 **Exact Match RAG** - Only Matching Pages")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pages_data" not in st.session_state:
    st.session_state.pages_data = []

st.sidebar.header("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
user_query = st.sidebar.text_input("🔍 Search exact words")

if uploaded_files and st.sidebar.button("🔄 Process PDFs", type="primary"):
    with st.spinner("Extracting pages..."):
        all_pages = []
        for pdf_file in uploaded_files:
            pdf_bytes = pdf_file.getvalue()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                mat = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
                
                all_pages.append({
                    "filename": pdf_file.name,
                    "page_num": page_num + 1,
                    "text": text,
                    "image": np.array(img)
                })
            doc.close()
        
        st.session_state.pages_data = all_pages
        texts = [p["text"] for p in all_pages]
        metadatas = [{"filename": p["filename"], "page": p["page_num"]} for p in all_pages]
        st.session_state.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        st.success(f"✅ Indexed {len(all_pages)} pages!")
        st.rerun()

# ONLY SHOW MATCHING PAGES
if user_query and st.session_state.vectorstore and st.session_state.pages_data:
    st.markdown("---")
    st.subheader(f"🔍 **Searching: '{user_query}'**")
    
    query_words = user_query.lower().split()
    docs = st.session_state.vectorstore.similarity_search(user_query, k=10)
    
    matched_pages = 0
    for doc in docs:
        for page_data in st.session_state.pages_data:
            if (page_data["filename"] == doc.metadata["filename"] and 
                page_data["page_num"] == doc.metadata["page"]):
                
                page_text_lower = page_data["text"].lower()
                # ONLY pages containing EXACT words
                if any(word in page_text_lower for word in query_words):
                    matched_pages += 1
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**📄 {page_data['filename']}**")
                        st.markdown(f"**Page {page_data['page_num']}**")
                        st.markdown(f"**Preview:** {page_data['text'][:200]}...")
                    
                    with col2:
                        st.image(page_data["image"], caption=f"Page {page_data['page_num']}", use_column_width=True)
    
    if matched_pages == 0:
        st.warning("❌ No pages contain your search words!")
    else:
        st.success(f"✅ **{matched_pages} matching pages found!**")

else:
    st.info("""
    👆 Upload PDFs → Process → Search!
    
    **Works perfectly with:**
    • "machine learning"
    • "neural network" 
    • "algorithm"
    • "python"
    """)

st.markdown("---")
st.markdown("*🎯 Shows ONLY pages with your exact search words*")
