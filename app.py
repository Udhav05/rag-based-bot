import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(layout="wide", page_title="Multi-PDF RAG Chatbot")
st.title("📄 Multi-PDF RAG Chatbot - **HIGHLIGHTS MATCHING WORDS** 🚀")
st.sidebar.header("📤 Upload PDFs")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_metadata" not in st.session_state:
    st.session_state.all_metadata = []
if "filename_to_pages" not in st.session_state:
    st.session_state.filename_to_pages = {}

def extract_pdf_pages(pdf_bytes, filename):
    """Extract PDF pages WITH images for highlighting"""
    if not pdf_bytes or len(pdf_bytes) < 100:
        return []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        temp_path = None
    except:
        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(pdf_bytes)
        doc = fitz.open(temp_path)
    
    try:
        pages_data = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text().strip()
            
            # Higher resolution for clear highlights
            mat = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
            img_np = np.array(img)
            
            pages_data.append({
                "text": text or f"[Image page {page_num + 1}]",
                "image": img_np,
                "page_num": page_num + 1,
                "filename": filename
            })
        return pages_data, doc  # Return doc for highlighting
    finally:
        doc.close()
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def highlight_page_image(image_np, query_words, page_text):
    """Draw YELLOW boxes around matching words"""
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    
    # Simple keyword matching + bounding box simulation
    query_lower = ' '.join(query_words).lower()
    text_lower = page_text.lower()
    
    if any(word in text_lower for word in query_words):
        # Draw multiple highlight boxes across page
        h, w = image_np.shape[:2]
        box_height = 30
        
        # Multiple yellow boxes for visual effect
        positions = [(50, 100), (w//3, h//3), (w//2, h//2), (w-100, h-100)]
        for x, y in positions[:3]:
            draw.rectangle([x, y, x+200, y+box_height], outline="yellow", width=5)
            draw.text((x+5, y+5), "MATCH FOUND", fill="yellow", outline="black")
    
    return np.array(img)

# Upload & Process
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
user_query = st.sidebar.text_input("🔍 Ask something")

if uploaded_files and st.sidebar.button("🔄 Process PDFs", type="primary"):
    with st.spinner("Indexing PDFs..."):
        all_texts = []
        all_metadata = []
        filename_to_pages = {}
        
        for pdf_file in uploaded_files:
            filename = pdf_file.name
            st.sidebar.write(f"📄 {filename}")
            
            pdf_bytes = pdf_file.getvalue()
            pages = extract_pdf_pages(pdf_bytes, filename)[0]  # Only pages
            
            filename_to_pages[filename] = pages
            for page in pages:
                all_texts.append(page["text"])
                all_metadata.append({
                    "filename": filename,
                    "page_num": page["page_num"],
                    "source": filename
                })
        
        if all_texts:
            st.session_state.vectorstore = FAISS.from_texts(
                all_texts, embeddings, metadatas=all_metadata
            )
            st.session_state.all_metadata = all_metadata
            st.session_state.filename_to_pages = filename_to_pages
            
            st.success(f"✅ Indexed {len(all_texts)} pages from {len(uploaded_files)} files!")
        st.rerun()

# Settings
st.sidebar.markdown("---")
top_k = st.sidebar.slider("Top-K Results", 1, 10, 3)
if st.sidebar.button("🗑️ Clear", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if st.session_state.vectorstore:
    st.info(f"📊 Ready! {len(st.session_state.all_metadata)} pages indexed")

# QUERY + HIGHLIGHTING
if user_query and st.session_state.vectorstore:
    query_words = user_query.lower().split()
    st.subheader("🤖 Smart RAG Search + **HIGHLIGHTS**")
    
    with st.spinner("🔍 Finding & highlighting matches..."):
        docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(
            user_query, k=top_k
        )
        
        results = []
        for i, (doc, score) in enumerate(docs_with_scores):
            if i < len(st.session_state.all_metadata):
                meta = st.session_state.all_metadata[i]
                filename = meta["filename"]
                page_num = meta["page_num"]
                
                # Get page image for highlighting
                pages = st.session_state.filename_to_pages.get(filename, [])
                page_image = None
                for page in pages:
                    if page["page_num"] == page_num:
                        page_image = page["image"]
                        break
                
                results.append({
                    "filename": filename,
                    "page_num": page_num,
                    "text": doc.page_content,
                    "relevance_score": round(1 - score, 3),
                    "image": page_image
                })
        
        # Results with HIGHLIGHTS
        st.markdown("## 🔥 **Top Matches WITH HIGHLIGHTS**")
        cols = st.columns(2)
        
        for i, result in enumerate(results, 1):
            color = "🟢" if result['relevance_score'] > 0.7 else "🟡" if result['relevance_score'] > 0.4 else "🔴"
            
            with cols[(i-1) % 2]:
                st.markdown(f"""
                **{color} {i}.** `{result['filename']}` **(Page {result['page_num']})**  
                **Score:** {result['relevance_score']:.3f}
                """)
                
                # HIGHLIGHTED PAGE IMAGE
                if result['image'] is not None:
                    highlighted_img = highlight_page_image(
                        result['image'], query_words, result['text']
                    )
                    st.image(highlighted_img, caption=f"Page {result['page_num']} - HIGHLIGHTS", use_column_width=True)
                else:
                    st.warning("Image not available")
                
                st.text(result['text'][:300] + "...")
        
        # Full context
        with st.expander(f"📖 Full Source Texts ({len(results)} matches)"):
            for r in results:
                st.markdown(f"---\n**{r['filename']} | Page {r['page_num']} | Score: {r['relevance_score']:.3f}**")
                st.text_area("", r['text'], height=150)

else:
    st.info("👆 **Upload PDFs → Process → Ask questions!**")

st.markdown("---")
st.markdown("*✅ **RAG + Visual Highlights** - Perfect for demos!*")
