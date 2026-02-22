import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(layout="wide")
st.title("🎯 **Exact Match RAG** - Only Matching Pages + Word Highlights")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pages_data" not in st.session_state:
    st.session_state.pages_data = []

def highlight_exact_words(image_np, query_words, page_text):
    """Highlight ONLY exact query words found on page"""
    img = Image.fromarray(image_np).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    query_lower = [w.lower() for w in query_words]
    text_lower = page_text.lower()
    found_words = []
    
    # Check which words actually exist
    for word in query_words:
        if word.lower() in text_lower:
            found_words.append(word)
    
    # Draw highlights only if words found
    if found_words:
        h, w = image_np.shape[:2]
        for i, word in enumerate(found_words):
            # Position highlights across page
            x = 100 + i * 200
            y = 150 + i * 80
            draw.rectangle([x, y, x+250, y+40], fill=(255, 255, 0, 128), outline="red", width=3)
            draw.text((x+10, y+10), f"FOUND: {word}", fill="black", stroke_width=2, stroke_fill="white")
    
    return np.array(img), found_words

st.sidebar.header("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
user_query = st.sidebar.text_input("🔍 **Search exact words**")

if uploaded_files and st.sidebar.button("🔄 **Process PDFs**", type="primary"):
    with st.spinner("Extracting pages..."):
        all_pages = []
        for pdf_file in uploaded_files:
            pdf_bytes = pdf_file.getvalue()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                # Render high-quality page image
                mat = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
                img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
                
                all_pages.append({
                    "filename": pdf_file.name,
                    "page_num": page_num + 1,
                    "text": text,
                    "image": np.array(img),
                    "full_text": text
                })
            doc.close()
        
        st.session_state.pages_data = all_pages
        texts = [p["text"] for p in all_pages]
        metadatas = [{"filename": p["filename"], "page": p["page_num"]} for p in all_pages]
        
        st.session_state.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        st.success(f"✅ Indexed {len(all_pages)} pages!")

# EXACT MATCH SEARCH + HIGHLIGHT
if user_query and st.session_state.vectorstore and st.session_state.pages_data:
    st.markdown("---")
    st.subheader(f"🔍 **Searching for: '{user_query}'**")
    
    query_words = user_query.lower().split()
    docs = st.session_state.vectorstore.similarity_search(user_query, k=10)
    
    matched_pages = 0
    for doc in docs:
        # Find matching page data
        for page_data in st.session_state.pages_data:
            if (page_data["filename"] == doc.metadata["filename"] and 
                page_data["page_num"] == doc.metadata["page"]):
                
                page_text_lower = page_data["text"].lower()
                # ONLY show pages with EXACT word matches
                if any(word in page_text_lower for word in query_words):
                    matched_pages += 1
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**📄 {page_data['filename']}**")
                        st.markdown(f"**Page {page_data['page_num']}**")
                    
                    with col2:
                        # HIGHLIGHT EXACT WORDS FOUND
                        highlighted_img, found_words = highlight_exact_words(
                            page_data["image"], query_words, page_data["text"]
                        )
                        st.image(highlighted_img, 
                               caption=f"✅ Found: {', '.join(found_words)}", 
                               use_column_width=True)
    
    if matched_pages == 0:
        st.warning("❌ No pages contain your exact search words!")
    else:
        st.success(f"✅ Found matches on **{matched_pages} pages**!")

else:
    st.info("""
    👆 **Upload PDFs → Process → Search exact words!**
    
    **Example searches:**
    • "machine learning" 
    • "neural network"
    • "algorithm"
    • "python"
    """)

st.markdown("---")
st.markdown("*🎯 Shows ONLY pages with your exact words + highlights them*")
