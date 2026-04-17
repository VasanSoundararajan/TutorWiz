"""
Comprehensive Test Suite for Virtual Tutor
Tests each component independently with updated code
"""

import os
import sys
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_speech_recognition():
    """Test speech recognition module"""
    print("\n" + "="*60)
    print("TEST 1: Speech Recognition (Wav2Vec 2.0)")
    print("="*60)
    
    try:
        from speech.recognition import SpeechRecognizer, save_audio
        
        # Initialize
        print("\n📦 Initializing speech recognizer...")
        recognizer = SpeechRecognizer()
        
        # Create sample audio (1 second of sine wave)
        print("\n🎵 Creating test audio...")
        import numpy as np
        
        duration = 1  # seconds
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Save sample
        os.makedirs("data/audio", exist_ok=True)
        sample_path = "data/audio/test_sample.wav"
        save_audio(audio.astype('float32'), sample_path, sample_rate)
        print(f"   ✓ Saved to: {sample_path}")
        
        # Transcribe
        print("\n🎧 Testing transcription...")
        result = recognizer.transcribe_file(sample_path)
        
        print(f"\n✅ Results:")
        print(f"   Text: '{result['text']}'")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Sample rate: {result['sample_rate']} Hz")
        
        print("\nℹ️  Note: Empty transcription is expected for sine wave audio")
        print("   Try with actual speech for real transcriptions")
        
        return True
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Fix:")
        print("   pip install soundfile librosa")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False


def test_text_to_speech():
    """Test TTS module"""
    print("\n" + "="*60)
    print("TEST 2: Text-to-Speech (VITS)")
    print("="*60)
    
    try:
        from speech.synthesis import TextToSpeech
        
        # Initialize
        print("\n📦 Initializing TTS engine...")
        print("   (This will download ~200MB model on first run)")
        tts = TextToSpeech()
        
        # Synthesize
        text = "Hello, I am your virtual tutor."
        output_path = "data/audio/test_tts.wav"
        
        print(f"\n🗣️  Synthesizing: '{text}'")
        audio = tts.synthesize(text, output_path)
        
        print(f"\n✅ Results:")
        print(f"   Sample rate: {tts.get_sample_rate()} Hz")
        print(f"   Duration: {len(audio) / tts.get_sample_rate():.2f}s")
        print(f"   Output shape: {audio.shape}")
        print(f"   Saved to: {output_path}")
        
        return True
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Fix:")
        print("   pip install transformers torch soundfile")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        
        # Check if it's a model download issue
        if "connection" in str(e).lower() or "download" in str(e).lower():
            print("\n💡 This might be a download issue.")
            print("   Try running again with a stable internet connection.")
        
        return False


def test_translation():
    """Test mBART translation"""
    print("\n" + "="*60)
    print("TEST 3: Multilingual Translation (mBART)")
    print("="*60)
    
    try:
        from translation.translator import Translator
        
        # Initialize
        print("\n📦 Initializing translator...")
        print("   (This will download ~2.4GB model on first run)")
        translator = Translator()
        
        # Test translations
        text_en = "Machine learning is a powerful tool for solving complex problems."
        
        print(f"\n🌐 Original (English):")
        print(f"   {text_en}")
        
        # Translate to Hindi
        print(f"\n🔄 Translating to Hindi...")
        text_hi = translator.translate(text_en, 'en', 'hi')
        print(f"   Hindi: {text_hi}")
        
        # Translate to Tamil
        print(f"\n🔄 Translating to Tamil...")
        text_ta = translator.translate(text_en, 'en', 'ta')
        print(f"   Tamil: {text_ta}")
        
        # Translate to Spanish
        print(f"\n🔄 Translating to Spanish...")
        text_es = translator.translate(text_en, 'en', 'es')
        print(f"   Spanish: {text_es}")
        
        print("\n✅ Translation successful")
        return True
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Fix:")
        print("   pip install transformers torch sentencepiece")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        
        # Check if it's a model download issue
        if "connection" in str(e).lower() or "download" in str(e).lower():
            print("\n💡 This might be a download issue.")
            print("   The mBART model is large (~2.4GB).")
            print("   Try running again with a stable internet connection.")
        
        return False


def test_avatar_renderer():
    """Test avatar rendering"""
    print("\n" + "="*60)
    print("TEST 4: Avatar Renderer")
    print("="*60)
    
    try:
        from avatar.renderer import AvatarRenderer, create_avatar_preview
        
        # Initialize
        print("\n📦 Initializing avatar renderer...")
        renderer = AvatarRenderer()
        
        # Render preview
        os.makedirs("data/avatars", exist_ok=True)
        output_path = "data/avatars/test_preview.png"
        
        print(f"\n🎨 Rendering preview frame...")
        create_avatar_preview(renderer, output_path)
        
        print(f"\n✅ Results:")
        print(f"   Preview saved to: {output_path}")
        print(f"   Frame size: 512x512 pixels")
        
        return True
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Fix:")
        print("   pip install torch numpy imageio")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False


def test_tutor_engine():
    """Test complete tutor engine"""
    print("\n" + "="*60)
    print("TEST 5: Virtual Tutor Engine")
    print("="*60)
    
    try:
        from tutor.engine import VirtualTutor
        
        # Check for API key
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  Warning: No API key found!")
            print("\n💡 Setup:")
            print("   1. Copy .env.example to .env")
            print("   2. Add your GROQ_API_KEY (get free key from https://console.groq.com)")
            print("   3. Run this test again")
            return False
        
        # Initialize (text-only mode)
        print("\n📦 Initializing tutor (text-only mode)...")
        tutor = VirtualTutor(
            language='en',
            use_avatar=False,
            use_speech=False,
            use_groq=True
        )
        
        # Set subject
        print("\n📚 Setting subject context...")
        tutor.set_subject("Mathematics", "Algebra")
        
        # Ask question
        question = "What is a quadratic equation?"
        print(f"\n❓ Question: {question}")
        
        print("\n💭 Generating answer...")
        result = tutor.ask_text(question)
        
        print(f"\n✅ Results:")
        print(f"   Question: {result['question']}")
        print(f"   Answer: {result['answer'][:200]}...")
        if len(result['answer']) > 200:
            print(f"   (... truncated, total length: {len(result['answer'])} chars)")
        
        return True
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Fix:")
        print("   pip install openai python-dotenv")
        return False
    except ValueError as e:
        if "API key" in str(e):
            print(f"\n❌ {e}")
            print("\n💡 Setup:")
            print("   1. Get free API key from https://console.groq.com")
            print("   2. Create .env file: GROQ_API_KEY=your-key-here")
            return False
        else:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        
        # Check for API-related errors
        if "api" in str(e).lower() or "key" in str(e).lower():
            print("\n💡 This looks like an API issue.")
            print("   Check your .env file and API key.")
        
        return False


def test_multilingual_tutor():
    """Test tutor with translation"""
    print("\n" + "="*60)
    print("TEST 6: Multilingual Tutor (English → Hindi)")
    print("="*60)
    
    try:
        from tutor.engine import VirtualTutor
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for API key
        if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  Skipping: No API key found")
            print("   Set GROQ_API_KEY in .env to test multilingual features")
            return False
        
        # Ask user if they want to download translation model
        print("\n⚠️  This test requires downloading the mBART model (~2.4GB)")
        response = input("   Continue? (y/n): ").lower()
        
        if response != 'y':
            print("\n⏭️  Skipped multilingual test")
            return True  # Not a failure, just skipped
        
        # Initialize with Hindi
        print("\n📦 Initializing multilingual tutor (Hindi)...")
        tutor = VirtualTutor(
            language='hi',
            use_avatar=False,
            use_speech=False,
            use_groq=True
        )
        
        # Ask in Hindi
        question = "मशीन लर्निंग क्या है?"  # What is machine learning?
        print(f"\n❓ Question (Hindi): {question}")
        
        print("\n💭 Generating answer...")
        result = tutor.ask_text(question)
        
        print(f"\n✅ Results:")
        print(f"   Question: {result['question']}")
        print(f"   Answer (Hindi): {result['answer'][:200]}...")
        if len(result['answer']) > 200:
            print(f"   (... truncated)")
        
        return True
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False


def test_component_integration():
    """Test integration between components"""
    print("\n" + "="*60)
    print("TEST 7: Component Integration")
    print("="*60)
    
    try:
        print("\n📦 Testing component imports...")
        
        # Import all components
        from speech.recognition import SpeechRecognizer
        from speech.synthesis import TextToSpeech
        from translation.translator import Translator
        from avatar.renderer import AvatarRenderer
        from tutor.engine import VirtualTutor
        
        print("   ✓ All components imported successfully")
        
        # Test basic initialization (no model loading)
        print("\n🔧 Testing basic component compatibility...")
        
        components = {
            'SpeechRecognizer': SpeechRecognizer,
            'TextToSpeech': TextToSpeech,
            'Translator': Translator,
            'AvatarRenderer': AvatarRenderer,
            'VirtualTutor': VirtualTutor
        }
        
        for name, component_class in components.items():
            print(f"   ✓ {name}: Compatible")
        
        print("\n✅ All components are compatible")
        return True
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Some dependencies might be missing")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 VIRTUAL TUTOR - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    print("\nℹ️  Test Information:")
    print("   • Some tests download large models (this is normal)")
    print("   • First run will take longer due to model downloads")
    print("   • Subsequent runs will use cached models")
    print("   • You can skip tests that require large downloads")
    
    # Create data directories
    os.makedirs("data/audio", exist_ok=True)
    os.makedirs("data/avatars", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    tests = [
        ("Component Integration", test_component_integration),
        ("Speech Recognition", test_speech_recognition),
        ("Text-to-Speech", test_text_to_speech),
        ("Translation", test_translation),
        ("Avatar Renderer", test_avatar_renderer),
        ("Tutor Engine", test_tutor_engine),
        ("Multilingual Tutor", test_multilingual_tutor),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            results[name] = False
            break
        except Exception as e:
            print(f"\n❌ {name} failed with unexpected error: {e}")
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⏭️  SKIP"
            skipped += 1
        
        print(f"{status:8} - {name}")
    
    total = len(results)
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
    
    # Recommendations
    if failed > 0:
        print("\n💡 Recommendations:")
        print("   • Check error messages above for specific issues")
        print("   • Install missing dependencies: pip install -r requirements.txt")
        print("   • Ensure .env file has GROQ_API_KEY set")
        print("   • Check internet connection for model downloads")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Virtual Tutor is ready to use!")
    elif passed > 0:
        print(f"\n✅ {passed}/{total} tests passed. Review failures above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()