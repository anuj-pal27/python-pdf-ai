# 🤖 AI Planet PDF Chatbot

A modern, intelligent PDF chatbot application that allows users to upload PDF documents and have natural conversations about their content using advanced AI technology.

![AI Planet PDF Chatbot](https://img.shields.io/badge/AI-Powered-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![React](https://img.shields.io/badge/React-18+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red) ![Groq](https://img.shields.io/badge/Groq-AI-purple)

## 🌟 Features

### 🚀 **AI-Powered Intelligence**
- **Groq Mixtral-8x7B** integration for lightning-fast responses
- **Smart document analysis** with vector embeddings
- **Contextual understanding** of PDF content
- **Detailed, comprehensive answers** (200-400+ words)

### 📄 **Document Processing**
- **PDF upload and extraction** using PyMuPDF
- **Text chunking and indexing** for optimal retrieval
- **Vector search** with FAISS for relevant content
- **Multi-document support** with session management

### 💬 **Chat Interface**
- **Real-time messaging** with WebSocket-like experience
- **Message history** persistence per document
- **Suggested questions** for quick start
- **Custom bot avatar** and professional UI
- **Mobile-responsive** design

### 🔧 **Advanced Features**
- **Fallback AI system** for reliability
- **Smart Q&A algorithms** when AI is unavailable
- **Professional formatting** with markdown support
- **Error handling** and graceful degradation

## 🏗 Architecture 

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI       │    │   Groq AI       │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (LLM)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │                       ▼                       │
        │              ┌─────────────────┐               │
        │              │   Vector DB     │               │
        │              │   (FAISS)       │               │
        │              └─────────────────┘               │
        │                       │                       │
        ▼                       ▼                       │
┌─────────────────┐    ┌─────────────────┐               │
│   Tailwind CSS  │    │   File Storage  │               │
│   (Styling)     │    │   (Local)       │               │
└─────────────────┘    └─────────────────┘               │
```

## 🛠 Tech Stack

### **Backend**
- **FastAPI** - Modern Python web framework
- **Groq AI** - Ultra-fast LLM inference
- **LangChain** - AI application development framework
- **FAISS** - Vector similarity search
- **SentenceTransformers** - Text embeddings
- **PyMuPDF** - PDF processing
- **Python 3.8+** - Core language

### **Frontend**
- **React 18** - Component-based UI
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Beautiful icons
- **React Icons** - Additional icon library

### **AI & ML**
- **Groq Mixtral-8x7B-32768** - Primary LLM
- **SentenceTransformers (all-MiniLM-L6-v2)** - Embeddings
- **FAISS CPU** - Vector database
- **Smart fallback algorithms** - Reliability layer

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** ([Download](https://python.org/downloads/))
- **Node.js 16 or higher** ([Download](https://nodejs.org/))
- **npm or yarn** (comes with Node.js)
- **Git** ([Download](https://git-scm.com/))

## 🚀 Quick Start

### 1. **Clone the Repository**

```bash
git clone https://github.com/anuj-pal27/python-pdf-ai.git
cd ai-planet-pdf-chatbot
```

### 2. **Get Your Groq API Key** 🔑

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a **free account**
3. Navigate to **API Keys** section
4. Create a new API key
5. Copy the key (you'll need it in step 4)

### 3. **Backend Setup** ⚙️

```bash
# Navigate to backend directory
cd backened

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. **Environment Configuration** 📝

Create a `.env` file in the `backened` directory:

```bash
# Create .env file
touch .env  # On Windows: type nul > .env
```

Add your API keys to the `.env` file:

```env
# Groq API Configuration (Required)
GROQ_API_KEY=your_groq_api_key_here

# Optional backup APIs
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### 5. **Frontend Setup** 🎨

```bash
# Navigate to frontend directory (new terminal)
cd pdf-summury

# Install dependencies
npm install
```

### 6. **Start the Application** 🚀

#### **Method 1: Automated Start (Recommended)**

```bash
# In backend directory
cd backened
chmod +x install_groq.sh
./install_groq.sh
```

#### **Method 2: Manual Start**

**Terminal 1 - Backend:**
```bash
cd backened
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd pdf-summury
npm run dev
```

### 7. **Access the Application** 🌐

- **Frontend**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📖 Usage Guide

### **1. Upload a PDF Document**
1. Click the **"Upload PDF"** button in the top-right corner
2. Select a PDF file from your computer
3. Wait for processing (usually 5-10 seconds)

### **2. Start Chatting**
Once uploaded, you can ask questions like:
- **"What is this document about?"**
- **"Summarize the main points in detail"**
- **"What skills and experience are mentioned?"**
- **"Describe the projects and their outcomes"**
- **"What is the educational background?"**

### **3. Advanced Features**
- **Suggested Questions**: Click on preset questions for quick start
- **Clear History**: Use the trash icon to reset conversations
- **Upload New Document**: Click the "+" button to switch documents
- **Detailed Responses**: Get 200-400+ word comprehensive answers

## 🔧 API Endpoints

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/` | GET | API information | `GET /` |
| `/health` | GET | System health check | `GET /health` |
| `/upload-pdf/` | POST | Upload & process PDF | `POST /upload-pdf/` |
| `/chat/` | POST | Send chat message | `POST /chat/` |
| `/ask-question/` | POST | One-time question | `POST /ask-question/` |
| `/chat-history/{id}` | GET | Get chat history | `GET /chat-history/abc123` |
| `/clear-chat/{id}` | POST | Clear history | `POST /clear-chat/abc123` |

### **Example API Call**

```javascript
// Chat with document
const response = await fetch('http://localhost:8000/chat/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    document_id: 'your-document-id',
    message: 'What are the key skills mentioned?'
  })
});

const data = await response.json();
console.log(data.answer);
```

## 📁 Project Structure

```
ai-planet-pdf-chatbot/
├── 📁 backened/                     # FastAPI Backend
│   ├── 📄 main.py                   # Main application
│   ├── 📄 requirements.txt          # Python dependencies
│   ├── 📄 .env                      # Environment variables
│   ├── 📄 install_groq.sh          # Setup script
│   ├── 📁 uploads/                 # PDF file storage
│   ├── 📁 vectorstores/            # Document embeddings
│   └── 📁 chat_histories/          # Chat sessions
│
├── 📁 pdf-summury/                 # React Frontend
│   ├── 📁 src/
│   │   ├── 📁 components/          # React components
│   │   │   ├── 📄 PdfUpload.jsx    # Main upload component
│   │   │   ├── 📄 Chatbot.jsx      # Chat interface
│   │   │   ├── 📄 ChatbotAvatar.jsx # Custom bot avatar
│   │   │   └── 📄 QuestionAnswer.jsx # Q&A component
│   │   ├── 📄 App.jsx              # Root component
│   │   ├── 📄 main.jsx             # Entry point
│   │   └── 📄 index.css            # Global styles
│   ├── 📄 package.json             # Dependencies
│   ├── 📄 tailwind.config.js       # Tailwind config
│   └── 📄 vite.config.js           # Vite config
│
└── 📄 README.md                    # This file
```

## 🎨 Customization

### **Change AI Model**
```python
# In backened/main.py
return ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name='llama2-70b-4096',  # Change model here
    max_tokens=2048
)
```

### **Modify Bot Avatar**
Edit `pdf-summury/src/components/ChatbotAvatar.jsx` to customize the bot's appearance.

### **Update Styling**
- **Colors**: Modify `tailwind.config.js`
- **Layout**: Edit component CSS files
- **Animations**: Update `src/index.css`

### **Add New Features**
- **Backend**: Add endpoints in `main.py`
- **Frontend**: Create new components in `src/components/`

## 🐛 Troubleshooting

### **Common Issues & Solutions**

#### **🔴 Backend Issues**

```bash
# Port already in use
lsof -ti:8000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8000   # Windows

# Dependencies issues
pip install --upgrade -r requirements.txt

# Virtual environment activation
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate          # Windows
```

#### **🔴 Frontend Issues**

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Port conflicts
npm run dev -- --port 3000

# Build issues
npm run build
```

#### **🔴 Groq API Issues**

1. **Verify API key** in `.env` file
2. **Check quota** at [console.groq.com](https://console.groq.com)
3. **Test connection**: Visit `http://localhost:8000/health`

#### **🔴 PDF Upload Issues**

- **File size**: Ensure PDF is under 10MB
- **File type**: Only PDF files are supported
- **Encoding**: Check for special characters in filename

### **Debug Mode**

Enable detailed logging:

```python
# Add to backened/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Deployment

### **Backend (Production)**

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### **Frontend (Production)**

```bash
# Build optimized version
npm run build

# Serve built files
npm run preview
```

### **Docker Deployment** (Optional)

```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY backened/ .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "main.py"]
```

## 🔒 Security Best Practices

- **🔐 API Keys**: Never commit `.env` files to version control
- **📁 File Upload**: Only PDF files are accepted and validated
- **🌐 CORS**: Configure allowed origins for production
- **⚡ Rate Limiting**: Implement for production usage
- **🛡️ Input Validation**: All inputs are sanitized and validated

## 📊 Performance

### **Response Times**
- **PDF Upload**: 5-15 seconds (depending on size)
- **Chat Response**: 1-3 seconds with Groq
- **Fallback Response**: <1 second

### **Supported File Sizes**
- **Maximum PDF Size**: 50MB
- **Optimal Size**: 1-10MB
- **Page Limit**: 100+ pages supported

### **Concurrent Users**
- **Development**: 5-10 users
- **Production**: 50+ users (with proper scaling)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Make** your changes
4. **Test** thoroughly
5. **Commit**: `git commit -am 'Add amazing feature'`
6. **Push**: `git push origin feature-name`
7. **Submit** a pull request

### **Development Guidelines**
- Follow **PEP 8** for Python code
- Use **ESLint** for JavaScript/React
- Add **tests** for new features
- Update **documentation** as needed

## 📝 Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `GROQ_API_KEY` | ✅ Yes | Groq API key | `gsk_...` |
| `OPENAI_API_KEY` | ❌ Optional | OpenAI fallback | `sk-...` |
| `HUGGINGFACEHUB_API_TOKEN` | ❌ Optional | HuggingFace fallback | `hf_...` |

## 📞 Support & Community

- **📚 Documentation**: This README + API docs at `/docs`
- **🐛 Bug Reports**: Create GitHub issues
- **💡 Feature Requests**: GitHub discussions
- **📧 Contact**: [Your email or contact info]

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **Groq** for lightning-fast AI inference
- **LangChain** for AI application framework
- **React Team** for the amazing frontend library
- **FastAPI** for the modern Python web framework
- **Tailwind CSS** for beautiful styling

---

## 🎯 Quick Commands Reference

```bash
# Backend setup
cd backened && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Frontend setup  
cd pdf-summury && npm install

# Start backend
cd backened && source venv/bin/activate && python main.py

# Start frontend
cd pdf-summury && npm run dev

# One-command start (recommended)
cd backened && ./install_groq.sh
```

---

**🎉 Happy Chatting with your PDFs!**

*Built with ❤️ for AI Planet*
