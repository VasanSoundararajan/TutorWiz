"""
FastAPI Backend for Virtual Tutor
Integrates with existing project structure
Uses NVIDIA GPT-120B + Svara TTS
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import uuid
import asyncio
from datetime import datetime
import shutil

# Add project root to path so 'src' can be imported
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import tutor engine
from src.tutor.engine_complete import VirtualTutor

# ============================================================
# Configuration
# ============================================================

app = FastAPI(
    title="Virtual Tutor API",
    description="AI-powered tutoring with NVIDIA GPT-120B and Svara TTS",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio", "responses")
SESSION_DIR = os.path.join(BASE_DIR, "data", "sessions")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

# Session storage
sessions: Dict[str, Dict] = {}


# ============================================================
# Pydantic Models
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    subject: Optional[str] = Field(None, description="Subject area")
    topic: Optional[str] = Field(None, description="Specific topic")
    session_id: Optional[str] = Field(None, description="Session ID")
    language: str = Field("en", description="Language code")
    include_audio: bool = Field(False, description="Generate audio response")
    include_video: bool = Field(False, description="Generate video response")
    use_avatar: bool = Field(False, description="Use avatar")


class QuestionResponse(BaseModel):
    answer: str
    session_id: str
    reasoning: Optional[str] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    timestamp: str


class SessionCreate(BaseModel):
    subject: Optional[str] = None
    topic: Optional[str] = None
    language: str = "en"
    use_speech: bool = False
    use_avatar: bool = False


class SessionResponse(BaseModel):
    session_id: str
    subject: Optional[str]
    topic: Optional[str]
    language: str
    created_at: str
    message_count: int


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


class TranscribeResponse(BaseModel):
    text: str
    duration: float
    language: str


class HealthResponse(BaseModel):
    status: str
    nvidia_api_available: bool
    speech_enabled: bool
    timestamp: str


# ============================================================
# Helper Functions
# ============================================================

def get_session(session_id: str) -> Dict:
    """Get or create session"""
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "tutor": None,
            "config": {}
        }
    return sessions[session_id]


def cleanup_old_files(directory: str, max_age_hours: int = 1):
    """Clean up old files"""
    current_time = datetime.now().timestamp()
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_time = os.path.getmtime(filepath)
        
        if current_time - file_time > (max_age_hours * 3600):
            try:
                os.remove(filepath)
                print(f"🗑️  Cleaned up: {filename}")
            except Exception as e:
                print(f"⚠️  Error cleaning {filename}: {e}")


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Virtual Tutor API - NVIDIA GPT-120B",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    nvidia_key = bool(os.getenv("NVIDIA_API_KEY"))
    
    return HealthResponse(
        status="healthy",
        nvidia_api_available=nvidia_key,
        speech_enabled=True,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create new tutoring session"""
    session_id = str(uuid.uuid4())
    
    session = get_session(session_id)
    session["config"] = {
        "subject": request.subject,
        "topic": request.topic,
        "language": request.language,
        "use_speech": request.use_speech,
        "use_avatar": request.use_avatar
    }
    
    # Initialize tutor for this session
    try:
        tutor = VirtualTutor(
            language=request.language,
            use_speech=request.use_speech,
            use_avatar=request.use_avatar
        )
        
        if request.subject:
            tutor.set_subject(request.subject, request.topic)
        
        session["tutor"] = tutor
        session["messages"] = []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize tutor: {str(e)}")
    
    return SessionResponse(
        session_id=session_id,
        subject=request.subject,
        topic=request.topic,
        language=request.language,
        created_at=session["created_at"],
        message_count=0
    )


@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session_info(session_id: str):
    """Get session info"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    config = session.get("config", {})
    messages = session.get("messages", [])
    
    return SessionResponse(
        session_id=session_id,
        subject=config.get("subject"),
        topic=config.get("topic"),
        language=config.get("language", "en"),
        created_at=session["created_at"],
        message_count=len(messages)
    )


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted", "session_id": session_id}


@app.post("/api/v1/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks
):
    """Ask tutor a question"""
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        session = get_session(session_id)
        
        # Initialize tutor if needed
        if session["tutor"] is None:
            tutor = VirtualTutor(
                language=request.language,
                use_speech=request.include_audio or request.include_video,
                use_avatar=request.use_avatar or request.include_video
            )
            
            if request.subject:
                tutor.set_subject(request.subject, request.topic)
            
            session["tutor"] = tutor
            session["messages"] = []
        else:
            tutor = session["tutor"]
        
        # Process question
        if request.include_video and request.use_avatar:
            # Generate video response
            video_filename = f"video_{uuid.uuid4()}.mp4"
            video_path = os.path.join(BASE_DIR, "data", "avatars", video_filename)
            
            result = tutor.generate_video_response(request.question, video_path)
            
            video_url = f"/api/v1/video/{video_filename}"
            audio_url = None
            
        elif request.include_audio:
            # Audio response only
            result = tutor.ask_text(request.question, stream=True)
            
            # Generate audio with Svara TTS
            if tutor.tts:
                from transformers import pipeline
                
                audio_result = tutor.tts(result['answer'])
                
                # Extract audio
                if isinstance(audio_result, dict):
                    audio = audio_result.get('audio', audio_result.get('waveform'))
                elif isinstance(audio_result, list):
                    audio = audio_result[0].get('audio')
                else:
                    audio = audio_result
                
                # Save audio
                import torch
                import numpy as np
                
                if torch.is_tensor(audio):
                    audio = audio.cpu().numpy()
                
                audio_filename = f"audio_{uuid.uuid4()}.wav"
                audio_path = os.path.join(AUDIO_DIR, audio_filename)
                
                import soundfile as sf
                sf.write(audio_path, audio, 16000)
                
                audio_url = f"/api/v1/audio/{audio_filename}"
                video_url = None
            else:
                audio_url = None
                video_url = None
        else:
            # Text only
            result = tutor.ask_text(request.question, stream=True)
            audio_url = None
            video_url = None
        
        # Store message
        if "messages" not in session:
            session["messages"] = []
        
        session["messages"].append({
            "role": "user",
            "content": request.question,
            "timestamp": datetime.now().isoformat()
        })
        session["messages"].append({
            "role": "assistant",
            "content": result['answer'],
            "timestamp": datetime.now().isoformat()
        })
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files, AUDIO_DIR)
        
        return QuestionResponse(
            answer=result['answer'],
            session_id=session_id,
            reasoning=result.get('reasoning'),
            audio_url=audio_url,
            video_url=video_url,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """Streaming response"""
    async def generate():
        try:
            session_id = request.session_id or str(uuid.uuid4())
            session = get_session(session_id)
            
            if session["tutor"] is None:
                tutor = VirtualTutor(language=request.language)
                if request.subject:
                    tutor.set_subject(request.subject, request.topic)
                session["tutor"] = tutor
            else:
                tutor = session["tutor"]
            
            # Build messages
            messages = [{"role": "system", "content": tutor.system_prompt}]
            messages.extend(tutor.conversation_history)
            messages.append({"role": "user", "content": request.question})
            
            # Stream
            for chunk in tutor.client.stream(messages):
                if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                    yield f"data: {{'type': 'reasoning', 'content': {repr(chunk.additional_kwargs['reasoning_content'])}}}\n\n"
                
                if chunk.content:
                    yield f"data: {{'type': 'content', 'content': {repr(chunk.content)}}}\n\n"
            
            yield f"data: {{'type': 'complete', 'session_id': '{session_id}'}}\n\n"
            
        except Exception as e:
            yield f"data: {{'type': 'error', 'message': {repr(str(e))}}}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/v1/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file"""
    try:
        # Save uploaded file
        temp_path = f"/tmp/{uuid.uuid4()}.wav"
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Transcribe
        from src.speech.recognition import SpeechRecognizer
        
        recognizer = SpeechRecognizer()
        result = recognizer.transcribe_file(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return TranscribeResponse(
            text=result['text'],
            duration=result['duration'],
            language="en"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech"""
    try:
        from transformers import pipeline
        
        # Initialize TTS
        tts = pipeline("text-to-speech", model="kenpath/svara-tts-v1")
        
        # Generate
        audio_result = tts(request.text)
        
        # Extract audio
        if isinstance(audio_result, dict):
            audio = audio_result.get('audio', audio_result.get('waveform'))
        elif isinstance(audio_result, list):
            audio = audio_result[0].get('audio')
        else:
            audio = audio_result
        
        # Save
        import torch
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        audio_filename = f"tts_{uuid.uuid4()}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        import soundfile as sf
        sf.write(audio_path, audio, 16000)
        
        return {
            "audio_url": f"/api/v1/audio/{audio_filename}",
            "text": request.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio file"""
    audio_path = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/api/v1/video/{filename}")
async def get_video(filename: str):
    """Serve video file"""
    video_path = os.path.join(BASE_DIR, "data", "avatars", filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(video_path, media_type="video/mp4")


@app.on_event("startup")
async def startup_event():
    """Startup"""
    print("="*60)
    print("🚀 Virtual Tutor API - NVIDIA Edition")
    print("="*60)
    print(f"\n   NVIDIA API: {'✓' if os.getenv('NVIDIA_API_KEY') else '✗ Not set'}")
    print(f"   Data dir: {BASE_DIR}/data/")
    print("\n✅ Ready at http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown"""
    print("\n👋 Shutting down...")
    cleanup_old_files(AUDIO_DIR)
