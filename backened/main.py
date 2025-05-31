from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
import uuid
import json
import numpy as np
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Free embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime


app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstores"
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Load embedding model once (smaller model for faster processing)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings

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
    # Generate embeddings
    embeddings = embedding_model.encode(text_chunks)
    # Create FAISS index
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(text_chunks, embeddings)),
        embedding=embedding_model
    )
    return vectorstore

def get_conversational_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def get_chat_history_path(document_id: str) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f"{document_id}.json")

def load_chat_history(document_id: str) -> List[Dict]:
    history_path = get_chat_history_path(document_id)
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    return []

def save_chat_history(document_id: str, history: List[Dict]):
    history_path = get_chat_history_path(document_id)
    with open(history_path, "w") as f:
        json.dump(history, f)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        doc = fitz.open(file_path)
        raw_text = get_pdf_text(doc)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)

        conversational = get_conversational_chain(vectorstore)
        document_id = str(uuid.uuid4())
        vs_path = os.path.join(VECTORSTORE_DIR, document_id)
        vectorstore.save_local(vs_path)

         # Initialize chat history
        welcome_msg = {
            "role": "assistant",
            "content": f"Hello! I'm ready to help you explore \"{file.filename}\".",
            "timestamp": datetime.now().isoformat()
        }
        save_chat_history(document_id, [welcome_msg])

        return {
            "filename": file.filename,
            "document_id": document_id,
            "num_chunks": len(text_chunks),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ask-question/")
async def chat(request: Request):
    try:
        data = await request.json()
        document_id = data.get("document_id")
        message = data.get("message")

        if not document_id or not message:
            raise HTTPException(400, "Missing document_id or message")
        
        #Load vectorstore
        vs_path = os.path.join(VECTORSTORE_DIR, document_id)
        if not os.path.exists(vs_path):
            raise HTTPException(404, "Document not found")
        
        vectorstore = FAISS.load_local(vs_path, embedding_model)
        
        # Load chat history
        chat_history = load_chat_history(document_id)

        # Create conversational chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Add previous messages to memory 
        for msg in chat_history[-4:]:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])
        # Initialize LLM
        llm = ChatOpenAI(temperature=0)

        qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = vectorstore.as_retriever(),
            memory = memory
        )

        # Get answer 
        result = qa({"question" : message})
        answer = result["answer"]

        # Update chat history
        new_msg =[
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
        updated_history = chat_history + new_msg
        save_chat_history(document_id, updated_history)

        return {
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
    
@app.get("/chat-history/{document_id}")
async def get_chat_history(document_id: str):
    try:
        history = load_chat_history(document_id)
        return {
            "history": history,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/clear-chat/{document_id}")
async def clear_chat_history(document_id: str):
    try:
        welcome_msg = {
            "role": "assistant",
            "content": "Chat history cleared! Ready to help again.",
            "timestamp": datetime.now().isoformat()
        }
        save_chat_history(document_id, [welcome_msg])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)