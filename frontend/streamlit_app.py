import streamlit as st
import requests
import os
import re
import tempfile
import pdfplumber

# Configuration
BACKEND_URL = "http://localhost:8000"  # Ensure this matches your FastAPI port
TEMP_DIR = "temp_media"

# Create temp directory if not exists
os.makedirs(TEMP_DIR, exist_ok=True)

st.set_page_config(page_title="Multimodal RAG Application", layout="wide")

def upload_pdf(file):
    """Upload PDF to backend and return processed result."""
    with st.spinner("Processing PDF..."):
        files = {"file": file}
        response = requests.post(f"{BACKEND_URL}/upload/pdf/", files=files)
        print(f"PDF Response: {response.status_code} - {response.text}")  # Debug print
        result = response.json()
        if response.status_code != 200:
            st.error(f"Backend error: {result.get('detail', 'Unknown error')}")
            return {}
        return result

def upload_video(file):
    """Upload video to backend and return processed result."""
    with st.spinner("Processing Video..."):
        files = {"file": file}
        response = requests.post(f"{BACKEND_URL}/upload/video/", files=files)
        print(f"Video Response: {response.status_code} - {response.text}")  # Debug print
        result = response.json()
        if response.status_code != 200:
            st.error(f"Backend error: {result.get('detail', 'Unknown error')}")
            return {}
        if 'full_transcript' in result:
            st.session_state.full_transcript = result['full_transcript']
        if 'transcript_chunks' in result:
            st.session_state.transcript_chunks = result['transcript_chunks']
        return result

def query_rag(question: str, context: str = None, source_type: str = None):
    """Query RAG system with optional context."""
    with st.spinner("Searching for answers..."):
        payload = {"question": question}
        if context:
            payload["context"] = context
            payload["source_type"] = source_type
        response = requests.post(f"{BACKEND_URL}/query/", json=payload)
        print(f"Query Response: {response.status_code} - {response.text}")  # Debug print
        return response.json()

def display_response(response):
    """Display the query response."""
    st.subheader("Answer")
    st.write(response['answer'])
    
    if 'sources' in response:
        st.subheader("Sources")
        for source in response['sources']:
            if source['type'] == "pdf":
                st.markdown(f"""
                **PDF Source**: {source['source']}  
                **Page**: {source['page']}  
                **Preview**: {source['content']}
                """)
            else:
                st.markdown(f"""
                **Video Source**: {source['source']}  
                **Timestamp**: {source['timestamp']}  
                **Preview**: {source['content']}
                """)
            st.divider()

def display_pdf_viewer(pdf_path: str):
    """Display PDF with text selection capability."""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"Page: {st.session_state.current_page + 1} of {total_pages}")
            new_page = st.number_input("Go to page", min_value=1, max_value=total_pages, 
                                     value=st.session_state.current_page + 1,
                                     key="pdf_page_nav")
            if new_page != st.session_state.current_page + 1:
                st.session_state.current_page = new_page - 1
                st.rerun()
        
        page = pdf.pages[st.session_state.current_page]
        img = page.to_image(resolution=150)
        
        if st.session_state.selected_text and st.session_state.selected_text.strip():
            words = st.session_state.selected_text.split()
            for word in words:
                escaped_word = re.escape(word.strip())
                if escaped_word:
                    matches = page.search(escaped_word)
                    for match in matches:
                        img.draw_rect(match, fill=(255, 255, 0, 64), stroke_width=2)
        
        st.image(img.original, use_container_width=True)
        
        st.subheader("Select Text for Query")
        page_text = page.extract_text()
        st.text_area("Page Text", value=page_text or "No text could be extracted", height=200, key="pdf_page_text")
        st.session_state.selected_text = st.text_input(
            "Highlight text (type or copy from above)", 
            value=st.session_state.selected_text,
            key="pdf_text_selection"
        )

def display_video_player(video_path: str):
    """Display video player with transcript segments and full transcription."""
    st.video(video_path, start_time=int(st.session_state.get('current_timestamp', 0) or 0))
    
    if 'transcript_chunks' not in st.session_state or not st.session_state.transcript_chunks:
        st.warning("Transcript data not available. Ensure video is processed by the backend.")
        return
    
    st.subheader("Full Transcript")
    st.text_area("Complete Transcribed Text", value=st.session_state.get('full_transcript', ""), height=200, key="full_transcript")
    
    st.subheader("Video Transcript Segments")
    for chunk in st.session_state.transcript_chunks:
        timestamp_parts = chunk['timestamp'].split(':')
        seconds = int(timestamp_parts[0]) * 60 + int(timestamp_parts[1])  # Convert "mm:ss" to seconds
        if st.button(f"{chunk['timestamp']}: {chunk['text'][:50]}...", key=f"transcript_{seconds}"):
            st.session_state.current_timestamp = seconds
            st.session_state.selected_transcript = chunk['text']
            st.rerun()
    
    if 'selected_transcript' in st.session_state:
        st.subheader("Selected Segment")
        st.text_area("Selected transcript segment", 
                    value=st.session_state.selected_transcript,
                    height=100,
                    key="selected_transcript_display")

def process_uploaded_files():
    """Handle file uploads and processing."""
    if 'pdf_file' in st.session_state and st.session_state.pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(st.session_state.pdf_file.getvalue())
            st.session_state.pdf_path = tmp_file.name
        result = upload_pdf(st.session_state.pdf_file)
        if 'file_id' in result:
            st.session_state.pdf_id = result['file_id']
            st.success(f"PDF processed successfully! File ID: {result['file_id']}")
        else:
            st.error(f"Failed to process PDF: {result.get('detail', 'No file_id in response')}")

    if 'video_file' in st.session_state and st.session_state.video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(st.session_state.video_file.getvalue())
            st.session_state.video_path = tmp_file.name
        result = upload_video(st.session_state.video_file)
        if 'file_id' in result:
            st.session_state.video_id = result['file_id']
            st.success(f"Video processing started! File ID: {result['file_id']}")
        else:
            st.error(f"Failed to process video: {result.get('detail', 'No file_id in response')}")

def reset_session():
    for key in list(st.session_state.keys()):
        if key in ['pdf_file', 'video_file', 'current_page', 'selected_text']:  # Preserve uploaders and page state
            continue
        if key.startswith('transcript') or key.startswith('pdf_id') or key.startswith('video_id'):
            del st.session_state[key]

def main():
    st.title("Multimodal RAG Application")
    st.markdown("Upload documents, select content, and ask questions.")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'selected_text' not in st.session_state:
        st.session_state.selected_text = ""
    if 'current_section' not in st.session_state:
        st.session_state.current_section = "pdf"
    
    # Navigation
    section = st.radio("Select Section", ["PDF", "Video"], key="section_radio", horizontal=True)
    st.session_state.current_section = section.lower()
    
    # File upload section
    with st.expander("Upload Files", expanded=True):
        if st.session_state.current_section == "pdf":
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Upload PDF")
                st.session_state.pdf_file = st.file_uploader(
                    "Choose a PDF file", 
                    type="pdf", 
                    key="pdf_uploader"
                )
            with col2:
                st.write("")  # Empty column to align layout
            if st.button("Process Uploaded Files", key="process_files_button"):
                process_uploaded_files()
        
        elif st.session_state.current_section == "video":
            col1, col2 = st.columns(2)
            with col2:
                st.subheader("Upload Video")
                st.session_state.video_file = st.file_uploader(
                    "Choose an MP4 file", 
                    type="mp4", 
                    key="video_uploader"
                )
            with col1:
                st.write("")  # Empty column to align layout
            if st.button("Process Uploaded Files", key="process_files_button"):
                process_uploaded_files()
    
    # Reset button
    with st.expander("Reset", expanded=False):
        if st.button("Clear Previous Results", key="reset_button"):
            reset_session()
            st.rerun()
    
    # Content Sections
    if st.session_state.current_section == "pdf" and 'pdf_path' in st.session_state:
        st.divider()
        st.subheader("PDF Viewer")
        display_pdf_viewer(st.session_state.pdf_path)
        if st.button("Ask About Selected Text", key="ask_pdf_button"):
            if st.session_state.selected_text:
                response = query_rag(
                    question="About the selected text",
                    context=st.session_state.selected_text,
                    source_type="pdf"
                )
                display_response(response)
    
    elif st.session_state.current_section == "video" and 'video_path' in st.session_state:
        st.divider()
        st.subheader("Video Player")
        display_video_player(st.session_state.video_path)
        if 'selected_transcript' in st.session_state and st.button(
            "Ask About Selected Segment", 
            key="ask_video_button"):
            response = query_rag(
                question="About the selected video segment",
                context=st.session_state.selected_transcript,
                source_type="video"
            )
            display_response(response)
    
    # General Query Section
    st.divider()
    st.subheader("General Query")
    question = st.text_input("Or ask a general question about the uploaded content:", 
                            key="general_question")
    if question:
        response = query_rag(question) 
        display_response(response)

if __name__ == "__main__":
    main()