// src/App.jsx
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Send, Settings, Volume2, VolumeX, Loader2 } from 'lucide-react';
import AnimatedAvatar from './components/AnimatedAvatar';
import { tutorAPI, SpeechRecognitionService } from './services/api';
import './App.css';

// Avatar data
const AVATARS = [
  {
    id: 'male',
    name: 'Alex',
    gender: 'male',
    image: '/avatars/male-avatar.png', // Replace with your male avatar
    description: 'Friendly male tutor'
  },
  {
    id: 'female',
    name: 'Sarah',
    gender: 'female',
    image: '/avatars/female-avatar.png', // Replace with your female avatar
    description: 'Patient female tutor'
  }
];

function App() {
  // State management
  const [selectedAvatar, setSelectedAvatar] = useState(AVATARS[0]);
  const [sessionId, setSessionId] = useState(null);
  const [subject, setSubject] = useState('');
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentWord, setCurrentWord] = useState('');
  const [isMuted, setIsMuted] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Refs
  const audioRef = useRef(null);
  const speechRecognition = useRef(null);
  const responseTextRef = useRef(null);

  // Initialize speech recognition
  useEffect(() => {
    speechRecognition.current = new SpeechRecognitionService();
    if (!speechRecognition.current.isSupported()) {
      console.warn('Speech recognition not supported');
    }
  }, []);

  // Check API status on mount
  useEffect(() => {
    checkAPIStatus();
  }, []);

  // Create session when avatar is selected
  useEffect(() => {
    if (selectedAvatar && !sessionId) {
      createSession();
    }
  }, [selectedAvatar]);

  // Auto-scroll response
  useEffect(() => {
    if (responseTextRef.current) {
      responseTextRef.current.scrollTop = responseTextRef.current.scrollHeight;
    }
  }, [response]);

  const checkAPIStatus = async () => {
    try {
      await tutorAPI.healthCheck();
      setApiStatus('connected');
      setError(null);
    } catch (err) {
      setApiStatus('error');
      setError('Cannot connect to backend. Please make sure the API server is running on http://localhost:8000');
    }
  };

  const createSession = async () => {
    try {
      const session = await tutorAPI.createSession(subject || null, 'en', true);
      setSessionId(session.session_id);
      console.log('Session created:', session.session_id);
    } catch (err) {
      setError('Failed to create session: ' + err.message);
    }
  };

  const startListening = () => {
    if (!speechRecognition.current.isSupported()) {
      setError('Speech recognition not supported in this browser');
      return;
    }

    setIsListening(true);
    setError(null);

    speechRecognition.current.startListening(
      (transcript) => {
        setQuestion(transcript);
        setIsListening(false);
        // Auto-submit after recognition
        setTimeout(() => handleSubmit(transcript), 500);
      },
      (error) => {
        setError('Speech recognition error: ' + error);
        setIsListening(false);
      }
    );
  };

  const stopListening = () => {
    speechRecognition.current.stopListening();
    setIsListening(false);
  };

  const handleSubmit = async (textQuestion = null) => {
    const finalQuestion = textQuestion || question;
    
    if (!finalQuestion.trim()) {
      setError('Please enter a question');
      return;
    }

    if (!sessionId) {
      setError('No active session. Please refresh the page.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResponse('');

    try {
      // Ask question
      const result = await tutorAPI.askQuestion(
        finalQuestion,
        sessionId,
        subject || null,
        true, // include audio
        'en'
      );

      // Set response text (remove thinking/reasoning if present)
      const cleanAnswer = result.answer.replace(/\[Thinking:.*?\]/g, '').trim();
      setResponse(cleanAnswer);

      // Play audio with lip sync
      if (result.audio_url && !isMuted) {
        await playAudioWithLipSync(result.audio_url, cleanAnswer);
      }

      // Clear question
      setQuestion('');
    } catch (err) {
      setError('Failed to get response: ' + err.message);
      console.error('Error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const playAudioWithLipSync = async (audioUrl, text) => {
    return new Promise((resolve, reject) => {
      const fullUrl = tutorAPI.getAudioUrl(audioUrl);
      
      if (audioRef.current) {
        audioRef.current.src = fullUrl;
        
        audioRef.current.onloadedmetadata = () => {
          setIsSpeaking(true);
          audioRef.current.play();

          // Sync words with audio
          const words = text.split(' ');
          const audioDuration = audioRef.current.duration;
          const wordDuration = audioDuration / words.length;

          let wordIndex = 0;
          const wordInterval = setInterval(() => {
            if (wordIndex < words.length) {
              setCurrentWord(words[wordIndex]);
              wordIndex++;
            } else {
              clearInterval(wordInterval);
            }
          }, wordDuration * 1000);

          audioRef.current.onended = () => {
            setIsSpeaking(false);
            setCurrentWord('');
            clearInterval(wordInterval);
            resolve();
          };
        };

        audioRef.current.onerror = (err) => {
          setIsSpeaking(false);
          setError('Failed to play audio');
          reject(err);
        };
      }
    });
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                Virtual Tutor AI
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                UMMATII COMPANY
              </p>
            </div>
            <div className="flex items-center space-x-4">
              {/* API Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  apiStatus === 'connected' ? 'bg-green-500' :
                  apiStatus === 'checking' ? 'bg-yellow-500' :
                  'bg-red-500'
                }`} />
                <span className="text-sm text-gray-600">
                  {apiStatus === 'connected' ? 'Connected' :
                   apiStatus === 'checking' ? 'Checking...' :
                   'Disconnected'}
                </span>
              </div>

              {/* Mute button */}
              <button
                onClick={() => setIsMuted(!isMuted)}
                className="p-2 rounded-full hover:bg-gray-100 transition"
              >
                {isMuted ? (
                  <VolumeX className="w-5 h-5 text-gray-600" />
                ) : (
                  <Volume2 className="w-5 h-5 text-gray-600" />
                )}
              </button>

              {/* Settings button */}
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 rounded-full hover:bg-gray-100 transition"
              >
                <Settings className="w-5 h-5 text-gray-600" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Avatar */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Your Tutor</h2>
            
            {/* Avatar Selection */}
            <div className="flex space-x-4 mb-6">
              {AVATARS.map((avatar) => (
                <button
                  key={avatar.id}
                  onClick={() => setSelectedAvatar(avatar)}
                  className={`flex-1 p-4 rounded-lg border-2 transition ${
                    selectedAvatar.id === avatar.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="text-center">
                    <div className="text-2xl mb-2">
                      {avatar.gender === 'male' ? '👨‍🏫' : '👩‍🏫'}
                    </div>
                    <div className="font-medium">{avatar.name}</div>
                    <div className="text-xs text-gray-500">{avatar.description}</div>
                  </div>
                </button>
              ))}
            </div>

            {/* Avatar Display */}
            <div className="aspect-square bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl overflow-hidden relative">
              <AnimatedAvatar
                avatar={selectedAvatar}
                isSpeaking={isSpeaking}
                currentWord={currentWord}
                gender={selectedAvatar.gender}
              />
            </div>

            {/* Subject Input */}
            {showSettings && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4"
              >
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Subject (optional)
                </label>
                <input
                  type="text"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="e.g., Mathematics, Physics, etc."
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </motion.div>
            )}
          </div>

          {/* Right Column - Interaction */}
          <div className="space-y-6">
            {/* Response Display */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-semibold mb-4">Response</h2>
              <div
                ref={responseTextRef}
                className="h-64 overflow-y-auto bg-gray-50 rounded-lg p-4 prose prose-sm max-w-none"
              >
                {isProcessing ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                    <span className="ml-3 text-gray-600">Thinking...</span>
                  </div>
                ) : response ? (
                  <div className="whitespace-pre-wrap">{response}</div>
                ) : (
                  <div className="text-gray-400 text-center h-full flex items-center justify-center">
                    Ask me anything! I'm here to help.
                  </div>
                )}
              </div>
            </div>

            {/* Input Area */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-semibold mb-4">Ask a Question</h2>
              
              {/* Error Display */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="space-y-4">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your question here..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows="4"
                  disabled={isProcessing || isListening}
                />

                <div className="flex space-x-3">
                  {/* Microphone Button */}
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={isListening ? stopListening : startListening}
                    disabled={isProcessing}
                    className={`flex-1 flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-medium transition ${
                      isListening
                        ? 'bg-red-500 hover:bg-red-600 text-white'
                        : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isListening ? (
                      <>
                        <MicOff className="w-5 h-5" />
                        <span>Stop Listening</span>
                      </>
                    ) : (
                      <>
                        <Mic className="w-5 h-5" />
                        <span>Voice Input</span>
                      </>
                    )}
                  </motion.button>

                  {/* Submit Button */}
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleSubmit()}
                    disabled={isProcessing || isListening || !question.trim()}
                    className="flex-1 flex items-center justify-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white rounded-lg font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <Send className="w-5 h-5" />
                        <span>Send</span>
                      </>
                    )}
                  </motion.button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Hidden Audio Element */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}

export default App;
