import React, { useState, useRef, useEffect } from 'react';
import { Send, MessageCircle, Trash2, FileText, X, Bot, User, Loader2, HelpCircle } from 'lucide-react';

const Chatbot = ({ documentId, documentName, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const suggestedQuestions = [
    "What is this document about?",
    "Summarize the main points",
    "Who are the authors?",
    "What are the key findings?",
    "What is the conclusion?"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    loadChatHistory();
  }, [documentId]);

  const loadChatHistory = async () => {
    try {
      const response = await fetch(`http://localhost:8000/chat-history/${documentId}`);
      if (response.ok) {
        const data = await response.json();
        const history = data.chat_history || [];
        setMessages(history);
        
        if (history.length === 0) {
          setMessages([{
            role: 'assistant',
            content: `Hello! I'm ready to help you explore "${documentName}". You can ask me questions about the document, or try one of the suggested questions below.`,
            timestamp: new Date().toLocaleTimeString()
          }]);
        }
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
      setMessages([{
        role: 'assistant',
        content: `Hello! I'm ready to help you explore "${documentName}". Ask me anything about the document!`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  const sendMessage = async (messageText = null) => {
    const messageToSend = messageText || inputMessage;
    if (!messageToSend.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: messageToSend,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document_id: documentId,
          message: messageToSend
        })
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMessage = {
          role: 'assistant',
          content: data.answer,
          timestamp: new Date().toLocaleTimeString()
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        const errorData = await response.json();
        const errorMessage = {
          role: 'assistant',
          content: `Sorry, I encountered an error: ${errorData.detail || 'Unknown error'}. Please try again.`,
          timestamp: new Date().toLocaleTimeString()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I cannot connect to the server. Please check your connection and try again.',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = async () => {
    try {
      const response = await fetch(`http://localhost:8000/clear-chat/${documentId}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setMessages([{
          role: 'assistant',
          content: `Chat history cleared! I'm ready to help you explore "${documentName}" again.`,
          timestamp: new Date().toLocaleTimeString()
        }]);
      }
    } catch (error) {
      console.error('Error clearing chat history:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSuggestedQuestion = (question) => {
    sendMessage(question);
  };

  return (
    <div className="h-full bg-white rounded-2xl shadow-xl border border-gray-100 flex flex-col">
      {/* Chat Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6 rounded-t-2xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
              <MessageCircle className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">AI Assistant</h3>
              <p className="text-blue-100 text-sm truncate max-w-48">
                {documentName}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={clearHistory}
              className="p-2 text-white hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
              title="Clear chat history"
            >
              <Trash2 className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-2 text-white hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
              title="Close chat"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
        {messages.map((message, index) => (
          <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
            <div className="flex items-end space-x-2 max-w-xs lg:max-w-md">
              {message.role === 'assistant' && (
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-blue-600" />
                </div>
              )}
              
              {/* Message Bubble */}
              <div className="flex flex-col">
                <div
                  className={`px-4 py-3 rounded-2xl shadow-sm ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white rounded-br-md'
                      : 'bg-white text-gray-800 border border-gray-200 rounded-bl-md'
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </p>
                </div>
                {message.timestamp && (
                  <p className={`text-xs mt-1 ${
                    message.role === 'user' ? 'text-right text-gray-400' : 'text-left text-gray-400'
                  }`}>
                    {message.timestamp}
                  </p>
                )}
              </div>

              {message.role === 'user' && (
                <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-gray-600" />
                </div>
              )}
            </div>
          </div>
        ))}
        
        {/* Suggested Questions (show only if few messages) */}
        {messages.length <= 1 && (
          <div className="space-y-3">
            <div className="flex items-center space-x-2 text-gray-500">
              <HelpCircle className="w-4 h-4" />
              <span className="text-sm">Try asking:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {suggestedQuestions.map((question, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestedQuestion(question)}
                  className="px-3 py-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-blue-50 hover:border-blue-200 hover:text-blue-700 transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}
        
        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex justify-start animate-fade-in">
            <div className="flex items-end space-x-2">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-blue-600" />
              </div>
              <div className="bg-white px-4 py-3 rounded-2xl rounded-bl-md border border-gray-200 shadow-sm">
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
                  <span className="text-sm text-gray-600">Thinking...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <div className="p-6 bg-white border-t border-gray-200 rounded-b-2xl">
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your document..."
              className="w-full p-4 pr-12 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
              rows="2"
              disabled={isLoading}
            />
            <div className="absolute bottom-2 right-2 text-xs text-gray-400">
              Press Enter to send
            </div>
          </div>
          <button
            onClick={() => sendMessage()}
            disabled={!inputMessage.trim() || isLoading}
            className="px-6 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center shadow-lg"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot; 