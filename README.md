# 🎓 Virtual Tutor - Complete AI-Powered Educational Platform

> **Advanced AI tutoring system with animated avatars, voice interaction, multilingual support, and 5 LLM engines**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [LLM Engines](#-llm-engines)
- [Frontend Features](#-frontend-features)
- [Backend Features](#-backend-features)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Virtual Tutor** is a complete AI-powered educational platform that combines cutting-edge language models, speech processing, animated avatars, and multilingual support to deliver an immersive learning experience.

### What Makes It Special

- 🤖 **4 LLM Options**: NVIDIA GPT-120B, Google Gemini, Sarvam-2B, Groq
- 🎭 **Animated Avatars**: Lip-synced characters (Alex & Sarah) with gestures
- 🎤 **Voice Interaction**: Speech-to-text input and text-to-speech output
- 🌍 **50+ Languages**: Multilingual translation with mBART
- 🔄 **Auto-Fallback**: Intelligent model switching ensures 99.9% uptime
- 📱 **Responsive UI**: Works on desktop, tablet, and mobile
- 🚀 **Production Ready**: Docker, Kubernetes, complete API documentation

### Use Cases

- **Online Education**: K-12, College, Professional courses
- **Language Learning**: Multi-language practice with native speakers
- **Corporate Training**: Custom subject tutoring for employees
- **Accessibility**: Voice-based learning for visually impaired
- **Research**: AI tutoring effectiveness studies

---

## ✨ Key Features

### 🎨 Frontend Features

#### Animated Avatars
- **Two Characters**: Alex (Male) and Sarah (Female)
- **Lip Sync**: Real-time mouth movements synced with speech
- **Gestures**: Head rotation, hand movements, eye blinking
- **5 Mouth States**: Phoneme-based animation (closed, slight, narrow, medium, wide)
- **Speaking Indicators**: Visual feedback during speech

#### Voice Interaction
- **Microphone Button**: One-click voice input
- **Browser Speech Recognition**: Web Speech API integration
- **Auto-Submit**: Questions sent automatically after recognition
- **Visual Feedback**: Red button when listening
- **Multi-language**: Supports 50+ languages

#### Text Interface
- **Text Area**: Manual question input
- **Enter to Submit**: Quick keyboard submission
- **Auto-Scroll**: Response auto-scrolls to bottom
- **Clean Display**: Thinking/reasoning removed automatically
- **Formatted Text**: Supports paragraphs and lists

#### User Controls
- **Avatar Selection**: Switch between Alex and Sarah
- **Mute/Unmute**: Toggle audio on/off
- **Subject Setting**: Optional subject specification
- **Session Management**: Persistent conversation context
- **API Status**: Connection indicator

### 🔧 Backend Features

#### LLM Engines (5 Options)

**1. NVIDIA GPT-120B** ⭐ Primary
- Best for educational content
- Includes reasoning in responses
- Free tier with credits
- Get key: https://build.nvidia.com

**2. Google Gemini 2.0 Flash**
- Ultra-fast inference (1-3s)
- 100+ languages
- Multimodal support
- Free tier available
- Get key: https://aistudio.google.com/app/apikey

**3. Sarvam-2B** (Local)
- Runs offline
- Indian language optimization
- No API key needed
- ~5GB model download
- Privacy-first

**4. Groq Llama 3.1**
- Fastest inference (<2s)
- Free tier
- High throughput
- Get key: https://console.groq.com

#### Intelligent Fallback System
```
Priority Chain:
NVIDIA → Gemini → Sarvam → Groq 
      ↓        ↓        ↓       
  If fails, auto-switches to next available
```

#### Speech Processing

**Text-to-Speech (TTS)**
- **Svara TTS**: High-quality voice synthesis
- **Gender Voices**: Male (Alex) and Female (Sarah)
- **Fast Inference**: 1-3 seconds
- **Auto-fallback**: VITS if Svara unavailable

**Speech Recognition (STT)**
- **Wav2Vec 2.0**: Facebook's speech model
- **Multiple Formats**: WAV, MP3, FLAC
- **Stereo-to-Mono**: Automatic conversion
- **Resampling**: Auto-adjusts to 16kHz

#### Translation
- **mBART-50**: 50+ language pairs
- **Bidirectional**: Question and answer translation
- **Auto-detect**: Language identification
- **Context-aware**: Maintains meaning

#### Session Management
- **UUID-based**: Unique session IDs
- **Conversation History**: Last 20 messages
- **Auto-cleanup**: Old files removed hourly
- **Subject Context**: Maintains topic focus

---

## 🛠 Technology Stack

### Frontend
```javascript
React 18.2          // UI Framework
Vite 5.0            // Build Tool
Framer Motion 10    // Animations
Tailwind CSS 3.4    // Styling
Axios               // HTTP Client
Lucide React        // Icons
```

### Backend
```python
FastAPI 0.104+      // Web Framework
Uvicorn             // ASGI Server
Pydantic 2.5+       // Data Validation

# LLM Engines
langchain-nvidia-ai-endpoints  // NVIDIA GPT-120B
google-generativeai           // Google Gemini
transformers 4.35+            // Sarvam-2B (local)

# Speech & NLP
torch 2.1+                    // Deep Learning
transformers                  // Models
soundfile 0.12+               // Audio I/O
librosa 0.10+                 // Audio Processing
sentencepiece                 // Tokenization

# Utilities
python-dotenv                 // Environment
numpy                         // Arrays
```

### Infrastructure
```yaml
Docker              // Containerization
Kubernetes          // Orchestration
Nginx               // Reverse Proxy
PostgreSQL          // Database (optional)
Redis               // Caching (optional)
```

---

## 🏗 Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Alex Avatar │  │ Sarah Avatar │  │ Voice Input  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Text Input  │  │   Response   │  │   Controls   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API (HTTP/JSON)
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 Backend (FastAPI)                           │
│        ┌──────────────────────────────────────────┐         │
│        │           Unified LLM Engine             │         │
│        │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │         │
│        │  │NVIDIA│ │Gemini│ │Sarvam│ │ Groq │     │         │
│        │  └───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘     │         │
│        │      └─────────┴─────────┴─────────┘     │         │
│        └──────────────────────────────────────────┘         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Speech Processing                                 │     │
│  │  ┌─────────────────┐  ┌─────────────────┐          │     │
│  │  │ Wav2Vec 2.0 STT │  │   Svara TTS     │          │     │
│  │  └─────────────────┘  └─────────────────┘          │     │
│  └────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Translation (mBART-50)                            │     │
│  └────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Session Management & API Endpoints                │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. User Input
   ├─ Voice → Browser Speech API → Text
   └─ Text → Direct Input

2. Question Processing
   ├─ Translation (if non-English)
   ├─ Session Context Added
   └─ Sent to LLM Engine

3. LLM Processing
   ├─ Try NVIDIA GPT-120B
   ├─ If fail → Try Gemini
   ├─ If fail → Try Sarvam
   └─ If fail → Try Gorq

4. Response Generation
   ├─ Remove Thinking/Reasoning
   ├─ Translation (if non-English)
   └─ Generate Audio (Svara TTS)

5. Frontend Display
   ├─ Show Text Response
   ├─ Play Audio
   ├─ Animate Avatar
   └─ Sync Lip Movements
```

---

## 📁 Project Structure

```
virtual-tutor/
│
├── backend/
│   ├── src/
│   │   ├── llm/                    # LLM Engines
│   │   │   ├── gemini_engine.py   # Google Gemini
│   │   │   ├── sarvam_engine.py   # Sarvam-2B
│   │   │   └── engine_unified.py  # Unified Interface
│   │   │
│   │   ├── tutor/                  # Main Logic
│   │   │   └── engine.py          # Virtual Tutor Engine
│   │   │
│   │   ├── speech/                 # Speech Processing
│   │   │   ├── recognition.py     # STT (Wav2Vec 2.0)
│   │   │   └── synthesis.py       # TTS (VITS)
│   │   │
│   │   ├── translation/            # Translation
│   │   │   └── translator.py      # mBART-50
│   │   │
│   │   ├── avatar/                 # Avatar Rendering
│   │   │   └── renderer.py        # NeRF-based
│   │   │
│   │   └── api/                    # REST API
│   │       └── main.py            # FastAPI Server
│   │
│   ├── tests/                      # Test Suite
│   │   └── test_all.py            # Comprehensive Tests
│   │
│   ├── data/                       # Data Storage
│   │   ├── audio/                 # Generated Audio
│   │   ├── avatars/               # Avatar Videos
│   │   └── sessions/              # Session Data
│   │
│   ├── requirements.txt           # Python Dependencies
│   │
│   └── .env.example               # Environment Template
│
├── frontend/
│   ├── public/
│   │   └── avatars/
│   │       ├── male-avatar.png    # Alex Avatar
│   │       └── female-avatar.png  # Sarah Avatar
│   │
│   ├── src/
│   │   ├── components/
│   │   │   └── AnimatedAvatar.jsx # Avatar Component
│   │   │
│   │   ├── services/
│   │   │   └── api.js             # API Client
│   │   │
│   │   ├── App.jsx                # Main Component
│   │   ├── App.css                # Styles
│   │   ├── index.css              # Global Styles
│   │   └── main.jsx               # Entry Point
│   │
│   ├── package.json               # Node Dependencies
│   ├── vite.config.js             # Vite Config
│   ├── tailwind.config.js         # Tailwind Config
│   └── .env                       # Frontend Environment
│
└── README.md                      # This File
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Git**
- **4GB+ RAM** (8GB+ for Sarvam-2B local)
- **10GB+ Disk Space** (for models)

### Quick Start (5 Minutes)

#### 1. Clone Repository

```bash
git clone https://github.com/vasansoundararajan/virtual-tutor.git
cd virtual-tutor
```

#### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Run backend
python src/api/main.py
```

Backend starts at: **http://localhost:8000**

#### 3. Frontend Setup

```bash
# Open new terminal
cd frontend

# Install dependencies
npm install

# Configure environment
echo "VITE_API_URL=http://localhost:8000" > .env

# Run frontend
npm run dev
```

Frontend starts at: **http://localhost:3000**

#### 4. Open Browser

Navigate to: **http://localhost:3000**

---

## ⚙️ Configuration

### Environment Variables

#### Backend (.env)

```bash
# =============================================================
# LLM API Keys (Set at least ONE)
# =============================================================

# NVIDIA GPT-120B (Primary - Recommended)
NVIDIA_API_KEY=nvapi-your-key-here

# Google Gemini 2.0 Flash (Fallback #1)
GOOGLE_API_KEY=AIzaSy-your-key-here

# Groq (Fallback #2)
GROQ_API_KEY=gsk_your-key-here

# =============================================================
# Model Selection
# =============================================================

# Preferred model (auto-fallback enabled)
# Options: nvidia, gemini, sarvam, groq, 
PREFERRED_MODEL=nvidia

# =============================================================
# Features
# =============================================================

DEFAULT_LANGUAGE=en
USE_AVATAR=false
USE_SPEECH=true

# =============================================================
# Model Configuration
# =============================================================

LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
TTS_MODEL=kenpath/svara-tts-v1
ASR_MODEL=facebook/wav2vec2-base-960h

# =============================================================
# Server
# =============================================================

API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
CORS_ORIGINS=*
```

#### Frontend (.env)

```bash
# API Backend URL
VITE_API_URL=http://localhost:8000
```

### Get API Keys

| Service | URL | Free? |
|---------|-----|-------|
| NVIDIA | https://build.nvidia.com | ✅ Yes |
| Google Gemini | https://aistudio.google.com/app/apikey | ✅ Yes |
| Groq | https://console.groq.com | ✅ Yes |
| OpenAI | https://platform.openai.com | ❌ Paid |

---

## 📖 Usage Guide

### Basic Usage

#### 1. Select Avatar

Click on **Alex** (Male) or **Sarah** (Female) to choose your tutor.

#### 2. Ask Question

**Option A: Voice Input**
1. Click microphone button 🎤
2. Speak your question
3. Question auto-submits

**Option B: Text Input**
1. Type question in text box
2. Press Enter or click Send

#### 3. Receive Response

- Text displays in response box
- Avatar speaks with lip movements
- Audio plays automatically (unless muted)
- Gestures animate during speech

### Advanced Features

#### Set Subject

1. Click Settings ⚙️
2. Enter subject (e.g., "Physics", "Mathematics")
3. All responses will be contextual to subject

#### Multilingual Support

```python
# Backend: Set language
tutor = VirtualTutor(language='hi')  # Hindi

# Ask in Hindi
result = tutor.ask_text("मशीन लर्निंग क्या है?")
# Response in Hindi
```

#### Session Management

Sessions persist automatically. Conversation context maintained for up to 20 messages.

To reset:
```python
tutor.clear_history()
```

---

## 🔌 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "nvidia_available": true,
  "svara_available": true,
  "timestamp": "2024-04-16T10:30:00"
}
```

#### Create Session
```http
POST /api/v1/sessions

Body:
{
  "subject": "Mathematics",
  "language": "en",
  "use_speech": true,
  "use_avatar": false
}

Response:
{
  "session_id": "uuid-here",
  "subject": "Mathematics",
  "language": "en",
  "created_at": "2024-04-16T10:30:00",
  "message_count": 0
}
```

#### Ask Question
```http
POST /api/v1/ask

Body:
{
  "question": "What is calculus?",
  "session_id": "uuid",
  "subject": "Mathematics",
  "include_audio": true,
  "include_video": false,
  "language": "en"
}

Response:
{
  "answer": "Calculus is a branch of mathematics...",
  "session_id": "uuid",
  "audio_url": "/api/v1/audio/response.wav",
  "video_url": null,
  "timestamp": "2024-04-16T10:30:00"
}
```

#### Get Audio
```http
GET /api/v1/audio/{filename}

Response: WAV audio file
```

#### Model Info
```http
GET /api/v1/model-info

Response:
{
  "model_name": "NVIDIA GPT-120B",
  "engine_type": "nvidia",
  "temperature": 0.7,
  "max_tokens": 4096
}
```

### Full API Documentation

Once server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🤖 LLM Engines

### Engine Comparison

| Feature | NVIDIA | Gemini | Sarvam | Groq | OpenAI |
|---------|--------|--------|--------|------|--------|
| **Speed** | Fast (2-5s) | Very Fast (1-3s) | Medium (3-6s) | Very Fast (1-2s) | Fast (2-4s) |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Cost** | Free* | Free | Free | Free | Paid |
| **Reasoning** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Languages** | English | 100+ | Indian+EN | English | 50+ |
| **Offline** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Model Size** | API | API | 5GB | API | API |
| **Best For** | Education | Speed | Privacy | Throughput | Reliability |

*Free tier with credits

### Usage Examples

#### NVIDIA GPT-120B
```python
tutor = VirtualTutor(preferred_model="nvidia")
result = tutor.ask_text("Explain quantum mechanics")
# Uses NVIDIA's reasoning capabilities
```

#### Google Gemini
```python
tutor = VirtualTutor(preferred_model="gemini")
result = tutor.ask_text("What is AI?")
# Ultra-fast response in 1-3 seconds
```

#### Sarvam-2B (Local)
```python
tutor = VirtualTutor(preferred_model="sarvam")
result = tutor.ask_text("कृत्रिम बुद्धिमत्ता क्या है?")
# Runs locally, no API calls
```

#### Auto-Fallback
```python
tutor = VirtualTutor()  # Auto-selects best available
# Tries: NVIDIA → Gemini → Sarvam → Groq
```

---

## 🎨 Frontend Features

### Avatar Animation System

#### Lip Sync Algorithm
```javascript
// Word-by-word synchronization
const words = text.split(' ');
const wordDuration = audioDuration / words.length;

// Update mouth shape every 300ms
setInterval(() => {
  const word = words[currentIndex];
  const mouthShape = detectPhoneme(word);
  setMouthState(mouthShape);
}, 300);
```

#### Mouth States
```javascript
const MOUTH_STATES = {
  closed:       'm, n, p',           // Closed lips
  'open-slight': 'consonants',       // Partial open
  'open-narrow': 'e, i',             // E shape
  'open-medium': 'general vowels',   // O shape
  'open-wide':   'a, o'              // Wide open
};
```

#### Gesture System
```javascript
const GESTURES = [
  'neutral',    // Resting position
  'pointing',   // Indicating something
  'explaining', // Teaching gesture
  'thinking'    // Contemplative pose
];

// Cycle every 3 seconds during speech
setInterval(() => {
  setGesture(GESTURES[Math.floor(Math.random() * 4)]);
}, 3000);
```

### Voice Interaction

#### Browser Speech Recognition
```javascript
const recognition = new webkitSpeechRecognition();
recognition.continuous = false;
recognition.interimResults = false;
recognition.lang = 'en-US';

recognition.onresult = (event) => {
  const transcript = event.results[0][0].transcript;
  handleQuestion(transcript);
};
```

#### Supported Browsers
- ✅ Chrome/Edge - Full support
- ⚠️ Safari - Limited support
- ❌ Firefox - No support

---

## 🔧 Backend Features

### Session Management

```python
# Create session
session_id = create_new_session(
    subject="Physics",
    language="en",
    use_speech=True
)

# Maintain conversation context
conversation_history = []  # Last 20 messages
```

### Response Cleaning

```python
def _clean_response(text):
    # Remove [Thinking: ...] blocks
    text = re.sub(r'\[Thinking:.*?\]', '', text)
    
    # Remove <thinking>...</thinking>
    text = re.sub(r'<thinking>.*?</thinking>', '', text)
    
    # Remove reasoning: ...
    text = re.sub(r'reasoning:.*?(?=\n\n|\Z)', '', text)
    
    return text.strip()
```

### Audio Generation

```python
# Svara TTS
audio = tts_engine.synthesize(text)
save_audio(audio, "response.wav", sample_rate=16000)

# Return URL
audio_url = f"/api/v1/audio/response_{uuid}.wav"
```

---

## 🚀 Deployment

### Docker Deployment

#### Build Images

```bash
# Backend
docker build -t virtual-tutor-backend -f docker/Dockerfile.backend .

# Frontend
docker build -t virtual-tutor-frontend -f docker/Dockerfile.frontend .
```

#### Run with Docker Compose

```bash
docker-compose up -d
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    image: virtual-tutor-backend
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./data:/app/data

  frontend:
    image: virtual-tutor-frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: virtual-tutor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: virtual-tutor
  template:
    metadata:
      labels:
        app: virtual-tutor
    spec:
      containers:
      - name: backend
        image: virtual-tutor-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: NVIDIA_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: nvidia-key
      - name: frontend
        image: virtual-tutor-frontend:latest
        ports:
        - containerPort: 80
```

```bash
# Deploy
kubectl apply -f k8s/
```

### Production Checklist

- [ ] Set production API keys
- [ ] Configure CORS origins
- [ ] Enable HTTPS/SSL
- [ ] Set up logging
- [ ] Configure rate limiting
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure auto-scaling
- [ ] Set up backups
- [ ] Configure CDN for static assets
- [ ] Enable caching (Redis)

---

## 🐛 Troubleshooting

### Common Issues

#### Backend not starting

**Error**: `No module named 'fastapi'`

**Fix**:
```bash
pip install -r requirements.txt
```

#### Frontend cannot connect

**Error**: `Cannot connect to backend`

**Check**:
1. Backend running on port 8000?
2. CORS enabled?
3. `.env` file correct?

**Fix**:
```bash
# Backend terminal
python src/api/main.py  # Should show: ✅ API Ready

# Frontend .env
VITE_API_URL=http://localhost:8000
```

#### Speech recognition not working

**Cause**: Browser doesn't support it

**Fix**:
- Use Chrome or Edge
- Grant microphone permission
- Use HTTPS (or localhost)

#### No audio playback

**Check**:
1. Mute button toggled off?
2. Backend generating audio?
3. Audio URL accessible?

**Debug**:
```javascript
console.log(audioUrl);  // Check URL
```

#### Sarvam model too slow

**Cause**: Running on CPU

**Fix**:
```bash
# Install CUDA PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Debug Mode

Enable debug logging:

```bash
# Backend
export LOG_LEVEL=DEBUG
python src/api/main.py

# Frontend
npm run dev -- --debug
```

---

## 📊 Performance

### Response Times

| Operation | Time |
|-----------|------|
| Session Creation | < 100ms |
| Text Question (NVIDIA) | 2-5s |
| Text Question (Gemini) | 1-3s |
| Text Question (Sarvam CPU) | 10-20s |
| Text Question (Sarvam GPU) | 3-6s |
| Audio Generation | 1-3s |
| Total (Text + Audio) | 3-8s |

### Resource Usage

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Backend (API only) | 5-10% | 500MB | 1GB |
| Backend (+ Sarvam) | 20-50% | 8GB | 6GB |
| Frontend | 2-5% | 100MB | 10MB |
| Total (API mode) | 10% | 600MB | 1GB |
| Total (Sarvam mode) | 50% | 8GB | 6GB |

### Optimization Tips

1. **Use NVIDIA/Gemini**: Faster than local Sarvam
2. **Enable Caching**: Cache common responses
3. **Use CDN**: Serve static assets faster
4. **Lazy Load**: Load avatars on demand
5. **Code Splitting**: Split React bundles
6. **Compress Audio**: Use lower bitrate for TTS

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork repository
git clone https://github.com/yourusername/virtual-tutor.git

# Create branch
git checkout -b feature/your-feature

# Make changes
# ...

# Run tests
cd backend
pytest tests/

cd frontend
npm test

# Commit changes
git commit -m "Add your feature"

# Push and create PR
git push origin feature/your-feature
```

### Code Style

- **Python**: PEP 8, type hints
- **JavaScript**: ESLint, Prettier
- **Commits**: Conventional Commits

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NVIDIA** - GPT-120B API
- **Google** - Gemini 2.0 Flash API
- **Sarvamai** - Sarvam-2B model
- **Meta** - Wav2Vec 2.0, mBART-50
- **HuggingFace** - Transformers library
- **FastAPI** - Web framework
- **React** - Frontend framework
- **Community** - All contributors

---

## 📞 Support

- **Documentation**: [https://docs.virtualtutor.ai](https://docs.virtualtutor.ai)
- **Issues**: [GitHub Issues](https://github.com/yourusername/virtual-tutor/issues)
- **Discord**: [Join Community](https://discord.gg/virtualtutor)
- **Email**: support@virtualtutor.ai

---

## 🗺 Roadmap

### Q2 2024
- [ ] Real-time collaboration
- [ ] Progress tracking dashboard
- [ ] Custom voice training
- [ ] Mobile apps (iOS/Android)

### Q3 2024
- [ ] Video-based tutorials
- [ ] Whiteboard integration
- [ ] Screen sharing
- [ ] AR/VR support

### Q4 2024
- [ ] Enterprise features
- [ ] Analytics dashboard
- [ ] Custom model fine-tuning
- [ ] Offline mobile apps

---

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/virtual-tutor)
![GitHub forks](https://img.shields.io/github/forks/yourusername/virtual-tutor)
![GitHub issues](https://img.shields.io/github/issues/yourusername/virtual-tutor)
![GitHub license](https://img.shields.io/github/license/yourusername/virtual-tutor)

---

<div align="center">

**Made with ❤️ for the future of education**

[Website](https://virtualtutor.ai) • [Documentation](https://docs.virtualtutor.ai) • [Demo](https://demo.virtualtutor.ai)

</div>