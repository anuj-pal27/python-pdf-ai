from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
import os
import uuid
import json
import numpy as np
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
import re

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# Directories
UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstores"
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pydantic models
class QuestionRequest(BaseModel):
    document_id: str
    question: str

class ChatRequest(BaseModel):
    document_id: str
    message: str

def get_pdf_text(pdf_docs) -> str:
    return "".join(page.get_text() for page in pdf_docs)

def get_text_chunks(text: str) -> List[str]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks: List[str]):
    embeddings = embedding_model.encode(text_chunks)
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(text_chunks, embeddings)),
        embedding=embedding_model
    )
    return vectorstore

def get_groq_llm():
    """Get Groq LLM with proper configuration"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("No Groq API key found")
            return None
            
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=1024,
            timeout=60,
            max_retries=3,
        )
    except Exception as e:
        print(f"Error initializing Groq LLM: {e}")
        return None

def answer_question_with_groq(question: str, context_chunks: List[str]) -> str:
    """Use Groq to answer questions with context"""
    try:
        llm = get_groq_llm()
        if not llm:
            raise Exception("Groq LLM not available")
        
        # Combine relevant chunks
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
        
        # Create a detailed prompt
        prompt = f"""
You are an AI assistant helping users understand a document. Based on the provided context, answer the user's question accurately and helpfully.

Context from the document:
{context}

User's Question: {question}

Instructions:
1. Answer based ONLY on the information provided in the context
2. If the answer isn't in the context, say "I don't have enough information in the document to answer that question"
3. Be specific and detailed in your response
4. If asked about a person, provide their details from the context
5. If asked about skills, projects, or experience, list them clearly
6. Keep your answer focused and relevant to the question

Answer:"""

        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"Groq error: {e}")
        return None

def smart_qa_fallback(question: str, text_chunks: List[str]) -> str:
    """Improved fallback Q&A with better logic"""
    question_lower = question.lower()
    
    # Extract keywords from question
    question_words = re.findall(r'\b\w+\b', question_lower)
    question_words = [word for word in question_words if len(word) > 2 and word not in ['what', 'how', 'why', 'when', 'where', 'who', 'the', 'and', 'for', 'are', 'you', 'can']]
    
    # Score chunks based on keyword matches
    chunk_scores = []
    for i, chunk in enumerate(text_chunks):
        chunk_lower = chunk.lower()
        score = 0
        
        # Count keyword matches
        for word in question_words:
            score += chunk_lower.count(word) * 2
        
        # Bonus for exact phrase matches
        if any(phrase in chunk_lower for phrase in question_words):
            score += 5
            
        chunk_scores.append((score, i, chunk))
    
    # Sort by score and get best matches
    chunk_scores.sort(reverse=True, key=lambda x: x[0])
    
    if chunk_scores[0][0] > 0:  # If we found relevant content
        best_chunks = [chunk for score, _, chunk in chunk_scores[:3] if score > 0]
        
        # Try Groq first
        groq_answer = answer_question_with_groq(question, best_chunks)
        if groq_answer:
            return groq_answer
        
        # Fallback to rule-based answers
        return generate_contextual_answer(question, best_chunks)
    else:
        return f"I couldn't find specific information about '{question}' in the document. Could you try asking about the main content or specific sections of the document?"

def generate_contextual_answer(question: str, chunks: List[str]) -> str:
    """Generate contextual answers based on question type"""
    question_lower = question.lower()
    relevant_text = " ".join(chunks)[:800]
    
    if any(word in question_lower for word in ['skills', 'programming', 'languages', 'technologies']):
        # Extract skills information
        skills_section = ""
        for chunk in chunks:
            if 'skills' in chunk.lower() or 'programming' in chunk.lower():
                skills_section = chunk
                break
        
        if skills_section:
            return f"Based on the document, here are the skills mentioned:\n\n{skills_section}"
        else:
            return "I found some technical content but couldn't identify a specific skills section in the document."
    
    elif any(word in question_lower for word in ['projects', 'work', 'experience', 'built', 'developed']):
        # Extract projects information
        projects_info = []
        for chunk in chunks:
            if any(word in chunk.lower() for word in ['project', 'developed', 'built', 'website', 'application']):
                projects_info.append(chunk)
        
        if projects_info:
            return f"Here are the projects mentioned in the document:\n\n" + "\n\n".join(projects_info[:2])
        else:
            return "I couldn't find specific project information in the document."
    
    elif any(word in question_lower for word in ['education', 'degree', 'school', 'college', 'university']):
        # Extract education information
        education_info = []
        for chunk in chunks:
            if any(word in chunk.lower() for word in ['education', 'b.tech', 'degree', 'school', 'college', 'university', 'cgpa', 'percentage']):
                education_info.append(chunk)
        
        if education_info:
            return f"Here's the education information from the document:\n\n" + "\n\n".join(education_info[:2])
        else:
            return "I couldn't find specific education information in the document."
    
    elif any(word in question_lower for word in ['contact', 'email', 'phone', 'linkedin', 'github']):
        # Extract contact information
        contact_info = []
        for chunk in chunks:
            if any(word in chunk.lower() for word in ['@', 'linkedin', 'github', '+91', 'email', 'phone']):
                contact_info.append(chunk)
        
        if contact_info:
            return f"Here's the contact information from the document:\n\n" + "\n\n".join(contact_info[:1])
        else:
            return "I couldn't find specific contact information in the document."
    
    elif any(word in question_lower for word in ['who', 'name', 'person']):
        # Extract personal information
        for chunk in chunks:
            if any(char.isupper() for char in chunk[:50]):  # Likely contains names
                return f"Based on the document, here's the personal information:\n\n{chunk[:300]}"
        
        return "I found some personal information but couldn't extract specific details."
    
    else:
        # General answer
        return f"Based on your question about '{question}', here's the most relevant information from the document:\n\n{relevant_text}"

def get_chat_history_path(document_id: str) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f"{document_id}.json")

def load_chat_history(document_id: str) -> List[Dict]:
    history_path = get_chat_history_path(document_id)
    try:
        if os.path.exists(history_path):
            with open(history_path, "r", encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

def save_chat_history(document_id: str, history: List[Dict]):
    history_path = get_chat_history_path(document_id)
    try:
        with open(history_path, "w", encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(400, detail="Only PDF files are allowed")
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from PDF
        pdf_document = fitz.open(file_path)
        text = get_pdf_text(pdf_document)
        pdf_document.close()
        
        if not text.strip():
            raise HTTPException(400, detail="Could not extract text from PDF")
        
        # Create text chunks
        text_chunks = get_text_chunks(text)
        
        # Save chunks for fallback
        chunks_path = os.path.join(VECTORSTORE_DIR, f"{document_id}_chunks.json")
        with open(chunks_path, "w", encoding='utf-8') as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=2)
        
        # Create and save vectorstore
        vectorstore = get_vectorstore(text_chunks)
        vs_path = os.path.join(VECTORSTORE_DIR, document_id)
        vectorstore.save_local(vs_path)
        
        # Initialize chat history
        welcome_msg = {
            "role": "assistant",
            "content": f"Hello! I've successfully processed your document '{file.filename}'. I can help you find information, answer questions, or provide summaries. What would you like to know?",
            "timestamp": datetime.now().isoformat()
        }
        save_chat_history(document_id, [welcome_msg])
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "text_length": len(text),
            "chunks_count": len(text_chunks),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        # Load vectorstore
        vs_path = os.path.join(VECTORSTORE_DIR, request.document_id)
        if not os.path.exists(vs_path):
            raise HTTPException(404, detail="Document not found")
        
        # Load text chunks for fallback
        chunks_path = os.path.join(VECTORSTORE_DIR, f"{request.document_id}_chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding='utf-8') as f:
                text_chunks = json.load(f)
            answer = smart_qa_fallback(request.question, text_chunks)
        else:
            answer = "I'm having trouble accessing the document content. Please try uploading the document again."
        
        return {
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error answering question: {str(e)}")

@app.post("/chat/")
async def chat(request: Request):
    try:
        data = await request.json()
        document_id = data.get("document_id")
        message = data.get("message")

        if not document_id or not message:
            raise HTTPException(400, detail="Missing document_id or message")
        
        # Load chat history
        chat_history = load_chat_history(document_id)

        # Load text chunks for processing
        chunks_path = os.path.join(VECTORSTORE_DIR, f"{document_id}_chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding='utf-8') as f:
                text_chunks = json.load(f)
            answer = smart_qa_fallback(message, text_chunks)
        else:
            answer = "I'm having trouble accessing the document content. Please try uploading the document again."

        # Update chat history
        new_messages = [
            {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            },
            {
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            }
        ]
        updated_history = chat_history + new_messages
        save_chat_history(document_id, updated_history)

        return {
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error in chat: {str(e)}")

@app.get("/chat-history/{document_id}")
async def get_chat_history_endpoint(document_id: str):
    try:
        history = load_chat_history(document_id)
        return {
            "chat_history": history,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error loading chat history: {str(e)}")

@app.post("/clear-chat/{document_id}")
async def clear_chat_history_endpoint(document_id: str):
    try:
        # Check if document exists
        vs_path = os.path.join(VECTORSTORE_DIR, document_id)
        if not os.path.exists(vs_path):
            raise HTTPException(404, detail="Document not found")
            
        welcome_msg = {
            "role": "assistant",
            "content": "Chat history cleared! I'm ready to help you explore this document again with Groq AI.",
            "timestamp": datetime.now().isoformat()
        }
        save_chat_history(document_id, [welcome_msg])
        return {
            "status": "success",
            "message": "Chat history cleared"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error clearing chat history: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    groq_status = "available" if get_groq_llm() is not None else "fallback_mode"
    
    return {
        "status": "healthy",
        "message": "PDF Chatbot API with Groq is running",
        "components": {
            "pdf_processor": "PyMuPDF",
            "embeddings": "SentenceTransformers",
            "llm": f"Groq Mixtral-8x7B ({groq_status})",
            "vector_store": "FAISS",
            "fallback": "Smart Q&A available"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF Chatbot API with Groq AI",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
        "ai_engine": "Groq Mixtral-8x7B-32768",
        "endpoints": {
            "upload": "/upload-pdf/",
            "chat": "/chat/",
            "ask_question": "/ask-question/",
            "chat_history": "/chat-history/{document_id}",
            "clear_chat": "/clear-chat/{document_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)