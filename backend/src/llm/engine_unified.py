"""
Unified LLM Engine with Intelligent Fallback
Supports: NVIDIA GPT-120B → Gemini 2.0 Flash → Sarvam-2B → Groq → OpenAI
Location: src/tutor/engine_unified.py
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

load_dotenv()


class UnifiedLLMEngine:
    """
    Unified LLM Engine with automatic fallback
    Priority: NVIDIA → Gemini → Sarvam → Groq → OpenAI
    """
    
    def __init__(
        self,
        preferred_model: str = "sarvam",  # nvidia, gemini, sarvam, groq, openai
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize unified engine with fallback
        
        Args:
            preferred_model: Preferred LLM ('nvidia', 'gemini', 'sarvam', 'groq', 'openai')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.preferred_model = preferred_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.active_engine = None
        self.active_model_name = None
        
        print("="*60)
        print("🤖 UNIFIED LLM ENGINE INITIALIZATION")
        print("="*60)
        
        # Try to initialize engines in order of preference
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available engines with fallback"""
        
        engines_to_try = [
            ('nvidia', self._init_nvidia),
            ('gemini', self._init_gemini),
            ('sarvam', self._init_sarvam),
            ('groq', self._init_groq),
            ('openai', self._init_openai),
        ]
        
        # Try preferred model first
        for name, init_func in engines_to_try:
            if name == self.preferred_model:
                if init_func():
                    return
        
        # Try remaining models
        for name, init_func in engines_to_try:
            if name != self.preferred_model:
                if init_func():
                    return
        
        raise RuntimeError(
            "No LLM engine available. Please set at least one API key:\n"
            "- NVIDIA_API_KEY (NVIDIA GPT-120B)\n"
            "- GOOGLE_API_KEY (Gemini 2.0 Flash)\n"
            "- GROQ_API_KEY (Groq)\n"
            "- OPENAI_API_KEY (OpenAI)\n"
            "Or ensure transformers is installed for Sarvam-2B"
        )
    
    def _init_nvidia(self) -> bool:
        """Try to initialize NVIDIA GPT-120B"""
        try:
            print("\n1️⃣  Trying NVIDIA GPT-120B...")
            
            if not os.getenv("NVIDIA_API_KEY"):
                print("   ⚠️  NVIDIA_API_KEY not set")
                return False
            
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            
            self.active_engine = ChatNVIDIA(
                model="openai/gpt-oss-120b",
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.active_model_name = "NVIDIA GPT-120B"
            self.engine_type = "nvidia"
            
            print(f"   ✓ {self.active_model_name} initialized")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
    
    def _init_gemini(self) -> bool:
        """Try to initialize Google Gemini"""
        try:
            print("\n2️⃣  Trying Google Gemini 2.0 Flash...")
            
            if not os.getenv("GOOGLE_API_KEY"):
                print("   ⚠️  GOOGLE_API_KEY not set")
                return False
            
            import sys
            sys.path.append(os.path.dirname(__file__))
            from gemini_engine import GeminiEngine
            
            self.active_engine = GeminiEngine(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.active_model_name = "Google Gemini 2.0 Flash"
            self.engine_type = "gemini"
            
            print(f"   ✓ {self.active_model_name} initialized")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
    
    def _init_sarvam(self) -> bool:
        """Try to initialize Sarvam-2B"""
        try:
            print("\n3️⃣  Trying Sarvam-2B (local)...")
            
            import sys
            sys.path.append(os.path.dirname(__file__))
            from sarvam_engine import SarvamEngine
            
            self.active_engine = SarvamEngine(
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 2048)  # Sarvam has lower limit
            )
            
            self.active_model_name = "Sarvam-2B"
            self.engine_type = "sarvam"
            
            print(f"   ✓ {self.active_model_name} initialized")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
    
    def _init_groq(self) -> bool:
        """Try to initialize Groq"""
        try:
            print("\n4️⃣  Trying Groq...")
            
            if not os.getenv("GROQ_API_KEY"):
                print("   ⚠️  GROQ_API_KEY not set")
                return False
            
            from openai import OpenAI
            
            self.active_engine = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
            
            self.model_name = "llama-3.1-8b-instant"
            self.active_model_name = "Groq Llama 3.1 8B"
            self.engine_type = "groq"
            
            print(f"   ✓ {self.active_model_name} initialized")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
    
    def _init_openai(self) -> bool:
        """Try to initialize OpenAI"""
        try:
            print("\n5️⃣  Trying OpenAI...")
            
            if not os.getenv("OPENAI_API_KEY"):
                print("   ⚠️  OPENAI_API_KEY not set")
                return False
            
            from openai import OpenAI
            
            self.active_engine = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            self.model_name = "gpt-3.5-turbo"
            self.active_model_name = "OpenAI GPT-3.5 Turbo"
            self.engine_type = "openai"
            
            print(f"   ✓ {self.active_model_name} initialized")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
    
    def generate(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """
        Generate response using active engine
        
        Args:
            messages: List of message dicts
            stream: Whether to stream (only supported by some engines)
            
        Returns:
            Generated text
        """
        if not self.active_engine:
            raise RuntimeError("No active engine initialized")
        
        print(f"💭 Generating with {self.active_model_name}...")
        
        try:
            if self.engine_type == "nvidia":
                return self._generate_nvidia(messages, stream)
            elif self.engine_type == "gemini":
                return self._generate_gemini(messages, stream)
            elif self.engine_type == "sarvam":
                return self._generate_sarvam(messages, stream)
            elif self.engine_type in ["groq", "openai"]:
                return self._generate_openai_compatible(messages, stream)
        except Exception as e:
            print(f"⚠️  Generation failed with {self.active_model_name}: {e}")
            raise
    
    def _generate_nvidia(self, messages: List[Dict[str, str]], stream: bool) -> str:
        """Generate using NVIDIA"""
        full_response = ""
        
        if stream:
            for chunk in self.active_engine.stream(messages):
                # Check for reasoning
                if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                    pass  # Skip reasoning in output
                
                if chunk.content:
                    full_response += chunk.content
        else:
            response = self.active_engine.invoke(messages)
            full_response = response.content
        
        return full_response
    
    def _generate_gemini(self, messages: List[Dict[str, str]], stream: bool) -> str:
        """Generate using Gemini"""
        return self.active_engine.generate(messages, stream=stream)
    
    def _generate_sarvam(self, messages: List[Dict[str, str]], stream: bool) -> str:
        """Generate using Sarvam"""
        return self.active_engine.generate(messages, stream=False)  # Sarvam doesn't support streaming
    
    def _generate_openai_compatible(self, messages: List[Dict[str, str]], stream: bool) -> str:
        """Generate using OpenAI-compatible API (Groq/OpenAI)"""
        response = self.active_engine.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        
        return response.choices[0].message.content
    
    def get_model_info(self) -> Dict:
        """Get information about active model"""
        return {
            "model_name": self.active_model_name,
            "engine_type": self.engine_type,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# ============================================================
# Example Usage
# ============================================================

def test_unified_engine():
    """Test unified engine"""
    print("\n" + "="*60)
    print("Testing Unified LLM Engine")
    print("="*60)
    
    # Initialize with auto-fallback
    engine = UnifiedLLMEngine(preferred_model="sarvam")
    
    # Get model info
    info = engine.get_model_info()
    print(f"\n📊 Active Model: {info['model_name']}")
    print(f"   Type: {info['engine_type']}")
    print(f"   Temperature: {info['temperature']}")
    print(f"   Max Tokens: {info['max_tokens']}")
    
    # Test generation
    print("\n📝 Testing Generation...")
    print("-" * 60)
    
    messages = [
        {"role": "system", "content": "You are a helpful tutor."},
        {"role": "user", "content": "What is calculus in simple terms?"}
    ]
    
    response = engine.generate(messages, stream=False)
    
    print(f"\n✓ Response received ({len(response)} chars)")
    print(f"\nResponse: {response[:300]}...")


if __name__ == "__main__":
    test_unified_engine()
