"""
Virtual Tutor Engine - Complete Integration
All LLMs: NVIDIA → Gemini → Sarvam → Groq → OpenAI
Location: src/tutor/engine.py (UPDATED)
"""

import os
import sys
from typing import Dict, Optional
from dotenv import load_dotenv
import numpy as np

# Add llm directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llm'))

from engine_unified import UnifiedLLMEngine

load_dotenv()


class VirtualTutor:
    """
    Complete Virtual Tutor System with All LLM Options
    Priority: NVIDIA GPT-120B → Gemini 2.0 → Sarvam-2B → Groq → OpenAI
    """
    
    def __init__(
        self,
        language: str = 'en',
        use_avatar: bool = False,
        use_speech: bool = False,
        subject: Optional[str] = None,
        preferred_model: str = "sarvam"  # nvidia, gemini, sarvam, groq, openai
    ):
        """
        Initialize Virtual Tutor
        
        Args:
            language: Primary language
            use_avatar: Enable avatar
            use_speech: Enable speech I/O
            subject: Subject area
            preferred_model: Preferred LLM engine
        """
        load_dotenv()
        
        print("="*60)
        print("🎓 VIRTUAL TUTOR INITIALIZATION")
        print("="*60)
        
        self.language = language
        self.use_avatar = use_avatar
        self.use_speech = use_speech
        self.subject = subject
        self.preferred_model = preferred_model
        
        # ===== LLM Setup =====
        print("\n1️⃣  Setting up AI engine...")
        self._setup_llm()
        
        # ===== Speech Setup =====
        if use_speech:
            print("\n2️⃣  Setting up speech processing...")
            self._setup_speech()
        else:
            self.speech_recognizer = None
            self.tts = None
            print("\n2️⃣  Speech processing: DISABLED")
        
        # ===== Translation Setup =====
        if language != 'en':
            print("\n3️⃣  Setting up translation...")
            self._setup_translation()
        else:
            self.translator = None
            print("\n3️⃣  Translation: NOT NEEDED (English only)")
        
        # ===== Avatar Setup =====
        if use_avatar:
            print("\n4️⃣  Setting up avatar...")
            self._setup_avatar()
        else:
            self.avatar_renderer = None
            self.avatar_animator = None
            print("\n4️⃣  Avatar: DISABLED")
        
        # Conversation history
        self.conversation_history = []
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        print("\n" + "="*60)
        print("✅ VIRTUAL TUTOR READY")
        print(f"   Active Model: {self.llm_engine.active_model_name}")
        print("="*60 + "\n")
    
    def _setup_llm(self):
        """Setup unified LLM with automatic fallback"""
        try:
            self.llm_engine = UnifiedLLMEngine(
                preferred_model=self.preferred_model,
                temperature=0.7,
                max_tokens=4096
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize any LLM: {e}")
    
    def _setup_speech(self):
        """Setup speech recognition and synthesis"""
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        # Speech Recognition
        try:
            from speech.recognition import SpeechRecognizer
            self.speech_recognizer = SpeechRecognizer()
        except Exception as e:
            print(f"   ⚠️  Speech recognition failed: {e}")
            self.speech_recognizer = None
        
        # Text-to-Speech - Try Svara first
        try:
            from transformers import pipeline
            self.tts = pipeline("text-to-speech", model="kenpath/svara-tts-v1")
            self.tts_type = "svara"
            print(f"   ✓ Svara TTS ready")
        except Exception as e:
            # Fallback to VITS
            try:
                from speech.synthesis import TextToSpeech
                self.tts = TextToSpeech()
                self.tts_type = "vits"
                print(f"   ✓ VITS TTS ready (fallback)")
            except Exception as e2:
                print(f"   ⚠️  TTS setup failed: {e2}")
                self.tts = None
                self.tts_type = None
    
    def _setup_translation(self):
        """Setup translation"""
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        try:
            from translation.translator import Translator
            self.translator = Translator()
            print(f"   ✓ Translation ready ({self.language})")
        except Exception as e:
            print(f"   ⚠️  Translation failed: {e}")
            self.translator = None
    
    def _setup_avatar(self):
        """Setup avatar rendering"""
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        try:
            from avatar.renderer import AvatarRenderer, AvatarAnimator
            self.avatar_renderer = AvatarRenderer()
            self.avatar_animator = AvatarAnimator(self.avatar_renderer)
            print(f"   ✓ Avatar ready")
        except Exception as e:
            print(f"   ⚠️  Avatar setup failed: {e}")
            self.avatar_renderer = None
            self.avatar_animator = None
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on subject"""
        base_prompt = """You are an expert tutor. Your role is to:
- Explain concepts clearly and simply
- Provide examples when helpful
- Break down complex topics into understandable parts
- Encourage students when they struggle
- Answer questions thoroughly but concisely"""
        
        if self.subject:
            base_prompt = f"You are an expert {self.subject} tutor. " + base_prompt
        
        return base_prompt
    
    def set_subject(self, subject: str, topic: Optional[str] = None):
        """Set learning subject/topic"""
        self.subject = subject
        context = f"You are tutoring {subject}"
        if topic:
            context += f", specifically {topic}"
        context += "."
        
        self.system_prompt = context + " " + self._build_system_prompt()
        print(f"📚 Subject set: {subject}" + (f" - {topic}" if topic else ""))
    
    def ask_text(self, question: str) -> Dict:
        """
        Ask question via text
        
        Args:
            question: Student's question
            
        Returns:
            dict: {'question', 'answer', 'answer_en', 'model_used'}
        """
        print(f"❓ Question: {question}")
        
        # Translate question to English if needed
        question_en = question
        if self.translator and self.language != 'en':
            question_en = self.translator.translate(question, self.language, 'en')
            print(f"   Translated to EN: {question_en}")
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": question_en})
        
        # Generate response
        answer_en = self.llm_engine.generate(messages, stream=False)
        
        # Remove thinking/reasoning if present
        answer_en = self._clean_response(answer_en)
        
        # Translate answer back if needed
        answer = answer_en
        if self.translator and self.language != 'en':
            answer = self.translator.translate(answer_en, 'en', self.language)
            print(f"   Translated to {self.language.upper()}")
        
        # Update history
        self.conversation_history.append({"role": "user", "content": question_en})
        self.conversation_history.append({"role": "assistant", "content": answer_en})
        
        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return {
            'question': question,
            'answer': answer,
            'answer_en': answer_en,
            'model_used': self.llm_engine.active_model_name
        }
    
    def _clean_response(self, text: str) -> str:
        """Remove thinking/reasoning content from response"""
        import re
        
        # Remove [Thinking: ...] blocks
        text = re.sub(r'\[Thinking:.*?\]', '', text, flags=re.DOTALL)
        
        # Remove <thinking>...</thinking> blocks
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        
        # Remove reasoning: ... blocks
        text = re.sub(r'reasoning:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()
    
    def ask_voice(self, audio_path: str = None) -> Dict:
        """Ask question via voice"""
        if not self.use_speech or not self.speech_recognizer:
            raise ValueError("Speech is disabled")
        
        # Transcribe question
        print("🎤 Processing voice input...")
        result = self.speech_recognizer.transcribe_file(audio_path)
        question = result['text']
        
        # Get text answer
        response = self.ask_text(question)
        
        # Synthesize speech
        if self.tts:
            print("🗣️  Generating voice response...")
            audio = self._synthesize_speech(response['answer'])
            
            # Save audio
            output_path = "data/audio/response.wav"
            os.makedirs("data/audio", exist_ok=True)
            
            if self.tts_type == "svara":
                self._save_svara_audio(audio, output_path)
            else:
                import soundfile as sf
                sf.write(output_path, audio, self.tts.get_sample_rate())
            
            response['audio_path'] = output_path
        
        return response
    
    def _synthesize_speech(self, text: str) -> np.ndarray:
        """Synthesize speech from text"""
        if self.tts_type == "svara":
            result = self.tts(text)
            if isinstance(result, dict):
                audio = result.get('audio', result.get('waveform'))
            elif isinstance(result, list):
                audio = result[0].get('audio', result[0].get('waveform'))
            else:
                audio = result
            
            import torch
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()
            
            return audio
        else:
            return self.tts.synthesize(text)
    
    def _save_svara_audio(self, audio: np.ndarray, filepath: str, sample_rate: int = 16000):
        """Save Svara audio to file"""
        try:
            import soundfile as sf
            sf.write(filepath, audio, sample_rate)
        except ImportError:
            from scipy.io import wavfile
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(filepath, sample_rate, audio_int16)
    
    def get_model_info(self) -> Dict:
        """Get information about active LLM"""
        return self.llm_engine.get_model_info()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🔄 Conversation history cleared")


# ============================================================
# Example Usage
# ============================================================

def main():
    """Example usage"""
    
    # Initialize with preferred model (auto-fallback enabled)
    tutor = VirtualTutor(
        language='en',
        use_avatar=False,
        use_speech=False,
        preferred_model="nvidia"  # Will fallback automatically
    )
    
    # Get model info
    info = tutor.get_model_info()
    print(f"\n📊 Using: {info['model_name']}")
    
    # Set subject
    tutor.set_subject("Machine Learning", "Neural Networks")
    
    # Ask questions
    questions = [
        "What is a neural network?",
        "How does backpropagation work?",
        "What is gradient descent?"
    ]
    
    for q in questions:
        result = tutor.ask_text(q)
        print(f"\n{'='*60}")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:200]}...")
        print(f"Model: {result['model_used']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
