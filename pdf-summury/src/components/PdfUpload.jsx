import React, { useState } from 'react'
import { Upload, FileText, MessageCircle, Loader2, X, CheckCircle, AlertCircle } from 'lucide-react'
import Chatbot from './Chatbot'
import './PdfUpload.css'
// Import manual CSS as fallback
import './Manual.css'

const PdfUpload = () => {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [showChat, setShowChat] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0]
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
    } else {
      alert('Please select a valid PDF file.')
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setFile(droppedFile)
    } else {
      alert('Please drop a valid PDF file.')
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/upload-pdf/', {
        method: 'POST',
        body: formData,
      })
      if (response.ok) {
        const result = await response.json()
        setUploadResult(result)
        setShowChat(true)
      } else {
        const errorData = await response.json()
        alert(`Upload failed: ${errorData.detail}`)
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert('Upload failed. Please check if the backend server is running.')
    } finally {
      setUploading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setUploadResult(null)
    setShowChat(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="text-center">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
              PDF Q&A Chatbot
            </h1>
            <p className="text-gray-600 text-lg">
              Upload your PDF document and start an intelligent conversation
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
          
          {/* Left Panel - Upload Section */}
          <div className="space-y-6">
            
            {/* Upload Card */}
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-6">
                <h2 className="text-2xl font-bold text-white flex items-center">
                  <Upload className="w-6 h-6 mr-3" />
                  Upload Document
                </h2>
                <p className="text-blue-100 mt-2">
                  Choose a PDF file to analyze and chat with
                </p>
              </div>

              <div className="p-6">
                {!uploadResult ? (
                  <div className="space-y-6">
                    {/* File Drop Zone */}
                    <div
                      className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
                        dragOver
                          ? 'border-blue-400 bg-blue-50'
                          : file
                          ? 'border-green-400 bg-green-50'
                          : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                      }`}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                    >
                      <input
                        type="file"
                        accept=".pdf"
                        onChange={handleFileSelect}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      />
                      
                      {file ? (
                        <div className="space-y-3">
                          <CheckCircle className="w-12 h-12 text-green-500 mx-auto" />
                          <div>
                            <p className="text-lg font-semibold text-green-700">
                              {file.name}
                            </p>
                            <p className="text-green-600">
                              {(file.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                          <div>
                            <p className="text-lg font-semibold text-gray-700">
                              Drop your PDF here
                            </p>
                            <p className="text-gray-500">
                              or click to browse files
                            </p>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Upload Button */}
                    <button
                      onClick={handleUpload}
                      disabled={!file || uploading}
                      className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-6 rounded-xl font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:from-blue-700 hover:to-purple-700 transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg"
                    >
                      {uploading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span>Processing PDF...</span>
                        </>
                      ) : (
                        <>
                          <Upload className="w-5 h-5" />
                          <span>Upload & Analyze</span>
                        </>
                      )}
                    </button>
                  </div>
                ) : (
                  /* Upload Success */
                  <div className="space-y-6">
                    <div className="flex items-center space-x-3 p-4 bg-green-50 rounded-xl border border-green-200">
                      <CheckCircle className="w-8 h-8 text-green-500 flex-shrink-0" />
                      <div className="flex-1">
                        <h3 className="font-semibold text-green-800">
                          Upload Successful!
                        </h3>
                        <p className="text-green-600 text-sm">
                          {uploadResult.filename}
                        </p>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <p className="text-gray-500">Document ID</p>
                        <p className="font-mono text-gray-800">{uploadResult.document_id}</p>
                      </div>
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <p className="text-gray-500">Text Length</p>
                        <p className="font-semibold text-gray-800">{uploadResult.text_length.toLocaleString()} chars</p>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <button
                        onClick={() => setShowChat(true)}
                        className="w-full bg-gradient-to-r from-green-600 to-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:from-green-700 hover:to-blue-700 transition-all duration-200 flex items-center justify-center space-x-2"
                      >
                        <MessageCircle className="w-5 h-5" />
                        <span>Start Chatting</span>
                      </button>
                      
                      <button
                        onClick={handleReset}
                        className="w-full bg-gray-500 text-white py-3 px-4 rounded-xl font-semibold hover:bg-gray-600 transition-all duration-200 flex items-center justify-center space-x-2"
                      >
                        <X className="w-4 h-4" />
                        <span>Upload Another PDF</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Document Summary Card */}
            {uploadResult && uploadResult.summary && (
              <div className="bg-white rounded-2xl shadow-xl border border-gray-100">
                <div className="bg-gradient-to-r from-green-500 to-blue-500 p-6">
                  <h3 className="text-xl font-bold text-white flex items-center">
                    <FileText className="w-5 h-5 mr-3" />
                    Document Summary
                  </h3>
                </div>
                <div className="p-6">
                  <div className="bg-gray-50 p-4 rounded-xl border border-gray-200">
                    <p className="text-gray-700 leading-relaxed">
                      {uploadResult.summary}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Features Card */}
            {!uploadResult && (
              <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-4">✨ Features</h3>
                <div className="space-y-3">
                  {[
                    'AI-powered question answering',
                    'Intelligent document analysis',
                    'Real-time chat interface',
                    'Persistent conversation history',
                    'Advanced text search fallback'
                  ].map((feature, index) => (
                    <div key={index} className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span className="text-gray-600">{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Chat Section */}
          <div className="lg:h-[700px]">
            {showChat && uploadResult ? (
              <Chatbot
                documentId={uploadResult.document_id}
                documentName={uploadResult.filename}
                onClose={() => setShowChat(false)}
              />
            ) : (
              <div className="bg-white rounded-2xl shadow-xl border border-gray-100 h-full flex items-center justify-center">
                <div className="text-center text-gray-500 p-8">
                  <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                    <MessageCircle className="w-12 h-12 text-gray-400" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-700 mb-3">
                    Ready to Chat
                  </h3>
                  <p className="text-gray-600 max-w-sm mx-auto leading-relaxed">
                    Upload a PDF document to start an intelligent conversation. 
                    Ask questions, get summaries, and explore your content with AI.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default PdfUpload 