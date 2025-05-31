import { useState } from 'react'
import { Send, FileText, RotateCcw, MessageCircle } from 'lucide-react'
import './QuestionAnswer.css'

const QuestionAnswer = ({ pdfFile, documentId, onNewDocument }) => {
  const [question, setQuestion] = useState('')
  const [conversations, setConversations] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmitQuestion = async (e) => {
    e.preventDefault()
    if (!question.trim()) return

    const userQuestion = question.trim()
    setQuestion('')
    setIsLoading(true)

    // Add user question to conversation
    const newConversation = [
      ...conversations,
      { type: 'question', content: userQuestion, timestamp: new Date() }
    ]
    setConversations(newConversation)

    try {
      const response = await fetch('http://localhost:8000/ask-question/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document_id: documentId,
          question: userQuestion
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to get answer')
      }

      const result = await response.json()
      
      setConversations([
        ...newConversation,
        { type: 'answer', content: result.answer, timestamp: new Date() }
      ])
    } catch (error) {
      console.error('Question error:', error)
      setConversations([
        ...newConversation,
        { 
          type: 'error', 
          content: error.message || 'Sorry, I encountered an error while processing your question. Please try again.',
          timestamp: new Date()
        }
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="question-answer">
      <div className="document-info">
        <div className="document-header">
          <FileText size={24} />
          <div>
            <h3>{pdfFile.name}</h3>
            <p>{(pdfFile.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        </div>
        <button onClick={onNewDocument} className="new-document-button">
          <RotateCcw size={16} />
          Upload New Document
        </button>
      </div>

      <div className="conversation-container">
        {conversations.length === 0 ? (
          <div className="welcome-message">
            <MessageCircle size={48} />
            <h3>Ask a question about your document</h3>
            <p>I can help you understand, summarize, or find specific information in your PDF.</p>
          </div>
        ) : (
          <div className="conversation-list">
            {conversations.map((item, index) => (
              <div key={index} className={`conversation-item ${item.type}`}>
                <div className="conversation-content">
                  <div className="message-text">{item.content}</div>
                  <div className="message-timestamp">
                    {formatTimestamp(item.timestamp)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {isLoading && (
          <div className="conversation-item answer loading">
            <div className="conversation-content">
              <div className="typing-indicator">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
              <div className="message-text">Thinking...</div>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmitQuestion} className="question-form">
        <div className="input-container">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your document..."
            className="question-input"
            disabled={isLoading}
          />
          <button 
            type="submit" 
            className="submit-button"
            disabled={!question.trim() || isLoading}
          >
            <Send size={20} />
          </button>
        </div>
      </form>
    </div>
  )
}

export default QuestionAnswer 