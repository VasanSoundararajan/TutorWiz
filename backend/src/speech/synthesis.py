"""
Text-to-Speech Synthesis
Converts tutor responses to natural-sounding speech
"""

import torch
from transformers import VitsModel, AutoTokenizer
import numpy as np
import soundfile as sf
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class TextToSpeech:
    """
    Text-to-Speech using VITS (Conditional Variational Autoencoder with Adversarial Learning)
    Lightweight and fast TTS for avatar voice
    """
    
    def __init__(self, model_name: str = "facebook/mms-tts-eng", device: str = None):
        """
        Initialize TTS engine
        
        Args:
            model_name: Hugging Face TTS model
                - "facebook/mms-tts-eng" (English)
                - "facebook/mms-tts-hin" (Hindi)
                - "facebook/mms-tts-tam" (Tamil)
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔊 Loading TTS model: {model_name}")
        print(f"   Device: {self.device}")
        
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"✓ TTS engine ready")
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            
        Returns:
            Audio waveform as numpy array
        """
        print(f"🗣️  Synthesizing: '{text[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate speech
        with torch.no_grad():
            output = self.model(**inputs).waveform
        
        # Convert to numpy
        waveform = output.squeeze().cpu().numpy()
        
        # Save if path provided
        if output_path:
            sf.write(output_path, waveform, samplerate=self.model.config.sampling_rate)
            print(f"✓ Saved audio to: {output_path}")
        
        return waveform
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of generated audio"""
        return self.model.config.sampling_rate


class MultilingualTTS:
    """
    Multilingual Text-to-Speech
    Automatically selects appropriate voice based on language
    """
    
    LANGUAGE_MODELS = {
        'en': 'facebook/mms-tts-eng',
        'hi': 'facebook/mms-tts-hin',
        'ta': 'facebook/mms-tts-tam',
        'es': 'facebook/mms-tts-spa',
        'fr': 'facebook/mms-tts-fra',
    }
    
    def __init__(self, default_language: str = 'en', device: str = None):
        """
        Initialize multilingual TTS
        
        Args:
            default_language: Default language code
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_language = default_language
        self.engines = {}  # Cache loaded engines
        
        print(f"🌍 Multilingual TTS initialized (default: {default_language})")
    
    def _get_engine(self, language: str) -> TextToSpeech:
        """Get or create TTS engine for language"""
        if language not in self.engines:
            model_name = self.LANGUAGE_MODELS.get(language, self.LANGUAGE_MODELS['en'])
            self.engines[language] = TextToSpeech(model_name, self.device)
        return self.engines[language]
    
    def synthesize(self, text: str, language: str = None, output_path: str = None) -> np.ndarray:
        """
        Synthesize text in specified language
        
        Args:
            text: Text to synthesize
            language: Language code (uses default if None)
            output_path: Optional save path
            
        Returns:
            Audio waveform
        """
        lang = language or self.default_language
        engine = self._get_engine(lang)
        return engine.synthesize(text, output_path)


# ============================================================
# Utility Functions
# ============================================================

def play_audio(audio: np.ndarray, sample_rate: int = 16000):
    """
    Play audio through speakers
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate in Hz
    """
    import sounddevice as sd
    print("🔊 Playing audio...")
    sd.play(audio, sample_rate)
    sd.wait()
    print("✓ Playback complete")


def audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate audio duration in seconds"""
    return len(audio) / sample_rate
