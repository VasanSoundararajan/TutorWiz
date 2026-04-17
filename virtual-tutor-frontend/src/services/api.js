// src/services/api.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds for TTS generation
});

// API Service
export const tutorAPI = {
  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Create session
  async createSession(subject = null, language = 'en', use_speech = true) {
    try {
      const response = await api.post('/api/v1/sessions', {
        subject,
        language,
        use_speech,
        use_avatar: false
      });
      return response.data;
    } catch (error) {
      console.error('Session creation failed:', error);
      throw error;
    }
  },

  // Get session info
  async getSession(sessionId) {
    try {
      const response = await api.get(`/api/v1/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Get session failed:', error);
      throw error;
    }
  },

  // Delete session
  async deleteSession(sessionId) {
    try {
      const response = await api.delete(`/api/v1/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Delete session failed:', error);
      throw error;
    }
  },

  // Ask question
  async askQuestion(question, sessionId, subject = null, includeAudio = true, language = 'en') {
    try {
      const response = await api.post('/api/v1/ask', {
        question,
        session_id: sessionId,
        subject,
        include_audio: includeAudio,
        include_video: false,
        language
      });
      return response.data;
    } catch (error) {
      console.error('Ask question failed:', error);
      throw error;
    }
  },

  // Get audio URL
  getAudioUrl(audioPath) {
    if (!audioPath) return null;
    return `${API_BASE_URL}${audioPath}`;
  },

  // Text to speech
  async textToSpeech(text, language = 'en') {
    try {
      const response = await api.post('/api/v1/tts', {
        text,
        language
      });
      return response.data;
    } catch (error) {
      console.error('TTS failed:', error);
      throw error;
    }
  }
};

// Speech Recognition Service
export class SpeechRecognitionService {
  constructor() {
    this.recognition = null;
    this.isListening = false;
    
    // Check browser support
    if ('webkitSpeechRecognition' in window) {
      this.recognition = new webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
      this.recognition = new SpeechRecognition();
    }

    if (this.recognition) {
      this.recognition.continuous = false;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-US';
    }
  }

  isSupported() {
    return this.recognition !== null;
  }

  startListening(onResult, onError) {
    if (!this.recognition) {
      onError(new Error('Speech recognition not supported'));
      return;
    }

    this.isListening = true;

    this.recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      onResult(transcript);
    };

    this.recognition.onerror = (event) => {
      this.isListening = false;
      onError(event.error);
    };

    this.recognition.onend = () => {
      this.isListening = false;
    };

    try {
      this.recognition.start();
    } catch (error) {
      this.isListening = false;
      onError(error);
    }
  }

  stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
    }
  }

  setLanguage(lang) {
    if (this.recognition) {
      this.recognition.lang = lang;
    }
  }
}

export default api;
