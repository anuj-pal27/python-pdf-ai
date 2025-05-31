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
            temperature=0.2,  # Slightly increased for more creative responses
            max_tokens=2048,  # Increased from 1024 to 2048
            timeout=90,       # Increased timeout for longer responses
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
        
        # Combine more relevant chunks for better context
        context = "\n\n".join(context_chunks[:5])  # Increased from 3 to 5 chunks
        
        # Create a more detailed prompt for comprehensive responses
        prompt = f"""
You are an AI assistant helping users understand a document thoroughly. Based on the provided context, give a comprehensive, detailed, and well-structured answer to the user's question.

Context from the document:
{context}

User's Question: {question}

Instructions for your response:
1. Provide a DETAILED and COMPREHENSIVE answer based on the information in the context
2. Structure your response with clear sections and bullet points when appropriate
3. Include specific examples, names, dates, numbers, and details from the document
4. If the question asks for a summary, provide a thorough overview with multiple paragraphs
5. If asked about skills or experience, list them comprehensively with details
6. If asked about projects, describe each project with objectives, technologies, and outcomes
7. For educational background, include institutions, degrees, dates, and relevant coursework
8. Always aim for 200-400 words minimum unless the content doesn't support it
9. Use professional language and organize information logically
10. If multiple aspects are relevant, address each one thoroughly
11. Include context and background information to make the answer complete
12. If the answer isn't fully available in the context, clearly state what information is missing

Provide a detailed, well-organized response:"""

        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"Groq error: {e}")
        return None

def smart_qa_fallback(question: str, text_chunks: List[str]) -> str:
    """Enhanced fallback Q&A with better logic and longer responses"""
    question_lower = question.lower()
    
    # Extract keywords from question
    question_words = re.findall(r'\b\w+\b', question_lower)
    question_keywords = [word for word in question_words if len(word) > 3]
    
    # Find relevant chunks with better scoring
    relevant_chunks = []
    chunk_scores = []
    
    for chunk in text_chunks:
        chunk_lower = chunk.lower()
        score = 0
        
        # Keyword matching with different weights
        for keyword in question_keywords:
            if keyword in chunk_lower:
                # Higher score for exact matches
                score += chunk_lower.count(keyword) * 3
            
            # Partial matching for related terms
            for word in chunk_lower.split():
                if keyword in word or word in keyword:
                    score += 1
        
        if score > 0:
            relevant_chunks.append(chunk)
            chunk_scores.append(score)
    
    # Sort by relevance score
    if relevant_chunks:
        sorted_chunks = [chunk for _, chunk in sorted(zip(chunk_scores, relevant_chunks), reverse=True)]
        top_chunks = sorted_chunks[:8]  # Increased from 5 to 8 chunks
    else:
        top_chunks = text_chunks[:8]
    
    # Generate comprehensive response based on question type
    if any(word in question_lower for word in ['summary', 'summarize', 'overview', 'about']):
        response = "## Document Summary\n\n"
        response += "Based on my analysis of the document, here's a comprehensive overview:\n\n"
        
        # Combine multiple chunks for a thorough summary
        combined_text = " ".join(top_chunks)
        
        # Extract key information patterns
        sentences = re.split(r'[.!?]+', combined_text)
        important_sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:12]
        
        response += "\n".join(f"• {sentence.strip()}" for sentence in important_sentences if sentence.strip())
        
        if len(response) < 200:
            response += f"\n\nAdditional Context:\n{' '.join(top_chunks[:3])}"
            
    elif any(word in question_lower for word in ['skill', 'competenc', 'abilit', 'expert', 'proficien']):
        response = "## Skills and Competencies\n\n"
        response += "Based on the document content, here are the identified skills and competencies:\n\n"
        
        skills_text = " ".join(top_chunks)
        # Look for skill-related patterns
        skill_indicators = re.findall(r'(?:skill|expert|proficient|experience|knowledge|competent|familiar).*?(?:[.!?]|$)', skills_text, re.IGNORECASE)
        
        for i, skill in enumerate(skill_indicators[:10], 1):
            response += f"**{i}.** {skill.strip()}\n\n"
            
        if len(skill_indicators) == 0:
            response += "The document contains the following relevant information about capabilities:\n\n"
            response += "\n\n".join(top_chunks[:4])
            
    elif any(word in question_lower for word in ['project', 'work', 'experience', 'job', 'role', 'position']):
        response = "## Professional Experience and Projects\n\n"
        response += "Here's a detailed breakdown of the professional experience and projects mentioned:\n\n"
        
        for i, chunk in enumerate(top_chunks[:6], 1):
            if len(chunk.strip()) > 50:
                response += f"### Project/Experience {i}:\n{chunk.strip()}\n\n"
                
    elif any(word in question_lower for word in ['education', 'degree', 'university', 'college', 'school', 'academic']):
        response = "## Educational Background\n\n"
        response += "Based on the document, here's the educational information:\n\n"
        
        for i, chunk in enumerate(top_chunks[:5], 1):
            if len(chunk.strip()) > 30:
                response += f"**Education {i}:** {chunk.strip()}\n\n"
                
    elif any(word in question_lower for word in ['contact', 'phone', 'email', 'address', 'location']):
        response = "## Contact Information\n\n"
        response += "Here's the contact information found in the document:\n\n"
        
        # Look for contact patterns
        contact_text = " ".join(top_chunks)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', contact_text)
        phones = re.findall(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', contact_text)
        
        if emails:
            response += f"**Email(s):** {', '.join(emails)}\n\n"
        if phones:
            response += f"**Phone(s):** {', '.join(phones)}\n\n"
            
        response += "**Additional Details:**\n"
        response += "\n".join(top_chunks[:3])
        
    else:
        # General detailed response
        response = f"## Response to: {question}\n\n"
        response += "Based on the document content, here's a comprehensive answer:\n\n"
        
        # Provide detailed context from multiple chunks
        for i, chunk in enumerate(top_chunks[:6], 1):
            if len(chunk.strip()) > 40:
                response += f"**Point {i}:** {chunk.strip()}\n\n"
    
    # Ensure minimum response length
    if len(response) < 300:
        response += f"\n\n**Additional Context:**\nFor more specific information, here are additional relevant details from the document:\n\n"
        response += "\n\n".join(top_chunks[:4])
    
    return response if response.strip() else "I found some information in the document, but I need a more specific question to provide a detailed answer. Could you please rephrase your question or ask about a specific aspect of the document?"

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

        # First try Groq for comprehensive answers
        answer = None
        chunks_path = os.path.join(VECTORSTORE_DIR, f"{document_id}_chunks.json")
        
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding='utf-8') as f:
                text_chunks = json.load(f)
                
            # Try Groq first for detailed responses
            vs_path = os.path.join(VECTORSTORE_DIR, document_id)
            if os.path.exists(vs_path):
                try:
                    vectorstore = FAISS.load_local(vs_path, embedding_model, allow_dangerous_deserialization=True)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Increased from 5 to 8
                    
                    # Get relevant context chunks
                    relevant_docs = retriever.get_relevant_documents(message)
                    context_chunks = [doc.page_content for doc in relevant_docs]
                    
                    # Try Groq for comprehensive answer
                    answer = answer_question_with_groq(message, context_chunks)
                    
                except Exception as groq_error:
                    print(f"Groq failed: {groq_error}")
                    answer = None
            
            # Fallback to enhanced smart Q&A if Groq fails
            if not answer:
                answer = smart_qa_fallback(message, text_chunks)
        else:
            answer = "I'm having trouble accessing the document content. Please try uploading the document again."

        # Ensure we have a substantial response
        if answer and len(answer) < 150:
            answer += "\n\nWould you like me to elaborate on any specific aspect or provide more details about a particular area?"

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