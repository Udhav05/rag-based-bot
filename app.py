import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(layout="wide", page_title="Multi-PDF RAG")
st.title("🎯 Multi-PDF RAG Chatbot - **Production Ready**")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_metadata" not in st.session_state:
    st.session_state.all_metadata = []
if "filename_to_pages" not in st.session_state:
    st.session_state.filename_to_pages = {}

st.sidebar.header("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
user_query = st.sidebar.text_input("🔍 Search (e.g. 'machine learning')")

if uploaded_files and st.sidebar.button("🔄 Process PDFs", type="primary"):
    with st.spinner("Indexing PDFs..."):
        all_texts = []
        all_metadata = []
        filename_to_pages = {}
        
        for pdf_file in uploaded_files:
            filename = pdf_file.name
            pdf_bytes = pdf_file.getvalue()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            pages_data = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                # Get page image
                mat = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
                img_np = np.array(img)
                
                pages_data.append({
                    "text": text or f"[Image page {page_num+1}]",
                    "image": img_np,
                    "page_num": page_num + 1
                })
            
            filename_to_pages[filename] = pages_data
            
            for page_data in pages_data:
                all_texts.append(page_data["text"])
                all_metadata.append({
                    "filename": filename,
                    "page_num": page_data["page_num"]
                })
            doc.close()
        
        st.session_state.vectorstore = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadata)
        st.session_state.all_metadata = all_metadata
        st.session_state.filename_to_pages = filename_to_pages
        st.rerun()
        st.success(f"✅ Indexed {len(all_texts)} pages!")

# Search Results
if user_query and st.session_state.vectorstore:
    st.subheader("🔍 **Top Matches**")
    
    docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(user_query, k=5)
    
    for i, (doc, score) in enumerate(docs_with_scores):
        if i >= len(st.session_state.all_metadata):
            continue
            
        meta = st.session_state.all_metadata[i]
        filename = meta["filename"]
        page_num = meta["page_num"]
        relevance = round(1 - score, 3)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"**📄 {filename}**")
            st.markdown(f"**Page {page_num}**")
            st.markdown(f"**Score: {relevance:.3f}**")
            st.markdown(f"**Preview:** {doc.page_content[:200]}...")
        
        with col2:
            # Show page image (no highlighting crash)
            pages = st.session_state.filename_to_pages.get(filename, [])
            page_image = None
            for page in pages:
                if page["page_num"] == page_num:
                    page_image = page["image"]
                    break
            
            if page_image is not None:
                st.image(page_image, caption=f"Page {page_num}", use_column_width=True)

else:
    st.info("👆 **Upload PDFs → Process → Search!**\n\n**Demo:** Try 'machine learning', 'neural network', 'algorithm'")

st.sidebar.markdown("---")
st.sidebar.markdown("*✅ FAISS semantic search + page images*")
