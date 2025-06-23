import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from datetime import datetime
from app.services.embedding import LocalEmbeddingFunction
import requests

app = FastAPI()

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = chromadb.PersistentClient(path="chroma_db")
ef = LocalEmbeddingFunction()
collection = client.get_or_create_collection(
    name="multimodal_rag",
    embedding_function=ef
)



class Query(BaseModel):
    question: str

class DocumentResponse(BaseModel):
    text: str
    metadata: dict

class VideoResponse(BaseModel):
    text: str
    metadata: dict

@app.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            documents = []
            metadatas = []
            ids = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    doc_id = f"{file_id}_page_{page_num + 1}"
                    documents.append(text)
                    metadatas.append({
                        "source": file.filename,
                        "page": page_num + 1,
                        "type": "pdf",
                        "file_id": file_id
                    })
                    ids.append(doc_id)
            
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
        
        return {"message": "PDF processed successfully", "file_id": file_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.mp4")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_duration = video.duration  # Get the total duration in seconds
        
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name
            audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                full_text = recognizer.recognize_google(audio_data)
            except sr.RequestError as e:
                raise HTTPException(status_code=500, detail=f"Speech recognition request failed: {str(e)}")
            except sr.UnknownValueError:
                full_text = ""
        
        os.unlink(audio_path)
        video.close()
        
        transcript_chunks = []
        if full_text:
            # Dynamic chunking based on duration (e.g., 10-second intervals)
            chunk_duration = 10  # seconds
            num_chunks = int(audio_duration / chunk_duration) + 1
            chunk_size = len(full_text) // num_chunks if num_chunks > 0 else len(full_text)
            
            documents = []
            metadatas = []
            ids = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(full_text))
                chunk_text = full_text[start_idx:end_idx].strip()
                if chunk_text:
                    timestamp = f"{i * chunk_duration}:00"  # e.g., "0:00", "10:00"
                    doc_id = f"{file_id}_chunk_{i + 1}"
                    documents.append(chunk_text)
                    metadatas.append({
                        "source": file.filename,
                        "timestamp": timestamp,
                        "type": "video",
                        "file_id": file_id
                    })
                    ids.append(doc_id)
                    transcript_chunks.append({"text": chunk_text, "timestamp": timestamp})
            
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        return {
            "message": "Video processed successfully",
            "file_id": file_id,
            "full_transcript": full_text,
            "transcript_chunks": transcript_chunks if transcript_chunks else []
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/")
async def query_rag(query: Query):
    try:
        results = collection.query(
            query_texts=[query.question],
            n_results=5
        )
        
        # Prepare context for LLM
        context = ""
        sources = []
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context += f"Source ({metadata['type']} from {metadata['source']}): {doc}\n\n"
            
            if metadata['type'] == "pdf":
                sources.append({
                    "type": "pdf",
                    "source": metadata['source'],
                    "page": metadata['page'],
                    "file_id": metadata['file_id'],
                    "content": doc[:200] + "..."
                })
            else:
                sources.append({
                    "type": "video",
                    "source": metadata['source'],
                    "timestamp": metadata['timestamp'],
                    "file_id": metadata['file_id'],
                    "content": doc[:200] + "..."
                })
        
        # Get response from Ollama instead of OpenRouter
        answer = get_ollama_response(query.question, context)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_ollama_response(question: str, context: str) -> str:
    """
    Get response from Ollama's Mistral model using /api/generate endpoint.
    """
    # Truncate context to avoid overwhelming the model (e.g., first 2000 characters)
    truncated_context = context[:2000] if len(context) > 2000 else context
    prompt = f"""
    You are an AI assistant answering based on the following context.
    Cite sources if mentioned in the context.

    Context:
    {truncated_context}

    Question: {question}

    Answer:
    """
    
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)  # Increased to 60 seconds
        response.raise_for_status()
        result = response.json()
        if "response" in result:
            return result["response"]
        else:
            raise ValueError("No 'response' key in Ollama API result")
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {str(e)} - Status Code: {e.response.status_code}"
    except requests.exceptions.ReadTimeout as e:
        return f"Request timed out after 60 seconds: {str(e)}. Try a shorter context or increase timeout."
    except Exception as e:
        return f"Failed to get response from Ollama: {str(e)}"