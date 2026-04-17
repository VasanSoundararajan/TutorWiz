"""
Speech Recognition using Wav2Vec 2.0
Converts voice input to text for tutor interaction
FIXED VERSION - Uses soundfile instead of torchaudio
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from typing import Optional, Union, Dict
import warnings
import os

warnings.filterwarnings('ignore')


class SpeechRecognizer:
    """
    Automatic Speech Recognition using Wav2Vec 2.0
    Supports real-time and file-based transcription
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str = None):
        """
        Initialize speech recognizer
        
        Args:
            model_name: Hugging Face model name
                - "facebook/wav2vec2-base-960h" (English, 360MB)
                - "facebook/wav2vec2-large-960h-lv60-self" (English, better quality, 1.2GB)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🎤 Loading speech recognition model: {model_name}")
        print(f"   Device: {self.device}")
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"✓ Speech recognizer ready")
    
    def transcribe_file(self, audio_path: str, language: str = "en") -> Dict[str, any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            language: Language code (currently supports 'en')
            
        Returns:
            dict: {
                'text': transcribed text,
                'sample_rate': audio sample rate,
                'duration': duration in seconds
            }
        """
        print(f"🎧 Transcribing: {audio_path}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio using soundfile
        try:
            import soundfile as sf
            
            # Read audio file
            waveform, sample_rate = sf.read(audio_path, dtype='float32')
            
            print(f"   Loaded: {len(waveform)} samples at {sample_rate}Hz")
            
        except ImportError:
            raise ImportError(
                "\n❌ soundfile is not installed!\n"
                "Install it with: pip install soundfile\n"
                "This is required for audio loading."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
        
        # Handle multi-channel audio (stereo -> mono)
        if len(waveform.shape) > 1:
            print(f"   Converting {waveform.shape[1]} channels to mono")
            waveform = np.mean(waveform, axis=1)
        
        # Resample to 16kHz if needed (Wav2Vec 2.0 requirement)
        if sample_rate != 16000:
            print(f"   Resampling from {sample_rate}Hz to 16000Hz")
            try:
                import librosa
                waveform = librosa.resample(
                    waveform, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
                sample_rate = 16000
            except ImportError:
                print("⚠️  Warning: librosa not installed. Cannot resample audio.")
                print("   Install with: pip install librosa")
                print("   Proceeding with original sample rate (may affect accuracy)")
        
        # Transcribe
        text = self._transcribe_waveform(waveform, sample_rate)
        
        return {
            'text': text,
            'sample_rate': sample_rate,
            'duration': len(waveform) / sample_rate
        }
    
    def transcribe_numpy(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe numpy audio array
        
        Args:
            audio_array: Audio as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Transcribed text
        """
        # Ensure audio is 1D
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            try:
                import librosa
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            except ImportError:
                print("⚠️  Warning: librosa not installed. Cannot resample.")
                print("   Install with: pip install librosa")
        
        return self._transcribe_waveform(audio_array, 16000)
    
    def _transcribe_waveform(self, waveform: np.ndarray, sample_rate: int) -> str:
        """
        Internal method to transcribe waveform
        
        Args:
            waveform: Audio waveform as numpy array (1D)
            sample_rate: Sample rate in Hz
            
        Returns:
            Transcribed text
        """
        # Ensure waveform is 1D
        if len(waveform.shape) > 1:
            waveform = waveform.flatten()
        
        # Normalize audio
        if waveform.max() > 0:
            waveform = waveform / np.abs(waveform).max()
        
        # Process audio
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        input_values = inputs.input_values.to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # Clean up transcription
        transcription = transcription.strip()
        
        return transcription
    
    def transcribe_stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio stream chunk (for real-time recognition)
        
        Args:
            audio_chunk: Audio chunk as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Transcribed text from chunk
        """
        return self.transcribe_numpy(audio_chunk, sample_rate)


class MultilingualSpeechRecognizer(SpeechRecognizer):
    """
    Multilingual speech recognition
    Extends base recognizer with multi-language support
    """
    
    LANGUAGE_MODELS = {
        'en': 'facebook/wav2vec2-base-960h',
        'hi': 'ai4bharat/indicwav2vec-hindi',  # Hindi
        'ta': 'ai4bharat/indicwav2vec-tamil',  # Tamil
        'es': 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',
        'fr': 'jonatasgrosman/wav2vec2-large-xlsr-53-french',
    }
    
    def __init__(self, language: str = 'en', device: str = None):
        """
        Initialize multilingual recognizer
        
        Args:
            language: Language code ('en', 'hi', 'ta', 'es', 'fr')
            device: 'cuda' or 'cpu'
        """
        model_name = self.LANGUAGE_MODELS.get(language, self.LANGUAGE_MODELS['en'])
        super().__init__(model_name=model_name, device=device)
        self.language = language
        print(f"🌍 Language: {language}")


# ============================================================
# Utility Functions
# ============================================================

def record_audio(duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio as numpy array
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "\n❌ sounddevice is not installed!\n"
            "Install it with: pip install sounddevice\n"
            "This is required for microphone recording."
        )
    
    print(f"🎙️  Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("✓ Recording complete")
    
    return audio.squeeze()


def save_audio(audio: np.ndarray, filepath: str, sample_rate: int = 16000):
    """
    Save audio array to file
    
    Args:
        audio: Audio waveform as numpy array
        filepath: Output file path
        sample_rate: Sample rate in Hz
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "\n❌ soundfile is not installed!\n"
            "Install it with: pip install soundfile\n"
            "This is required for saving audio."
        )
    
    sf.write(filepath, audio, sample_rate)
    print(f"✓ Saved audio to: {filepath}")


def load_audio(filepath: str, target_sr: int = 16000) -> tuple:
    """
    Load audio file and resample if needed
    
    Args:
        filepath: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        (waveform, sample_rate) tuple
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "\n❌ soundfile is not installed!\n"
            "Install it with: pip install soundfile"
        )
    
    # Load audio
    waveform, sample_rate = sf.read(filepath, dtype='float32')
    
    # Convert stereo to mono if needed
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Resample if needed
    if sample_rate != target_sr:
        try:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        except ImportError:
            print(f"⚠️  Warning: Cannot resample from {sample_rate}Hz to {target_sr}Hz")
            print("   Install librosa: pip install librosa")
    
    return waveform, sample_rate


# ============================================================
# Test & Example
# ============================================================

def test_speech_recognition():
    """Test speech recognition with sample audio"""
    print("\n" + "="*60)
    print("Testing Speech Recognition")
    print("="*60)
    
    # Create test audio directory
    os.makedirs("data/audio", exist_ok=True)
    
    # Create a simple test audio file (1 second of silence)
    print("\n1. Creating test audio...")
    test_audio = np.zeros(16000, dtype='float32')  # 1 second at 16kHz
    save_audio(test_audio, "data/audio/test.wav", sample_rate=16000)
    
    # Initialize recognizer
    print("\n2. Initializing speech recognizer...")
    recognizer = SpeechRecognizer()
    
    # Transcribe
    print("\n3. Transcribing test audio...")
    result = recognizer.transcribe_file("data/audio/test.wav")
    
    print(f"\n✓ Results:")
    print(f"  Text: '{result['text']}'")
    print(f"  Duration: {result['duration']:.2f}s")
    print(f"  Sample rate: {result['sample_rate']}Hz")
    
    print("\n" + "="*60)
    print("✅ Speech recognition test complete!")
    print("="*60)
    
    # Note about empty transcription
    if not result['text']:
        print("\nℹ️  Note: Empty transcription is normal for silence/noise.")
        print("   Try with actual speech audio for real transcriptions.")


if __name__ == "__main__":
    test_speech_recognition()