"""
Google Gemini Engine - Gemini 2.0 Flash
Uses Google AI Studio API for text generation
Location: src/llm/gemini_engine.py
"""

import os
from typing import Dict, List, Optional, Iterator
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed")
    print("   Install: pip install google-generativeai")


class GeminiEngine:
    """
    Google Gemini 2.0 Flash Engine
    High-speed reasoning with multimodal capabilities
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        top_k: int = 40
    ):
        """
        Initialize Gemini engine
        
        Args:
            api_key: Google AI Studio API key
            model: Model name (gemini-2.0-flash-exp, gemini-1.5-pro, etc.)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter. Get key from: https://aistudio.google.com/app/apikey"
            )
        
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        print(f"🚀 Initializing Google Gemini: {model}")
        
        # Configure API
        genai.configure(api_key=self.api_key)
        
        # Create model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'max_output_tokens': self.max_tokens,
            }
        )
        
        # Create chat session
        self.chat = None
        
        print(f"✓ Gemini engine ready")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> str:
        """
        Generate response from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response
            
        Returns:
            Generated text response
        """
        if stream:
            return self._generate_stream(messages)
        else:
            return self._generate_sync(messages)
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert OpenAI-style messages to Gemini format"""
        # Separate system message and conversation
        system_message = ""
        conversation = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                system_message = content
            elif role == 'user':
                conversation.append({
                    'role': 'user',
                    'parts': [content]
                })
            elif role == 'assistant':
                conversation.append({
                    'role': 'model',  # Gemini uses 'model' instead of 'assistant'
                    'parts': [content]
                })
        
        return system_message, conversation
    
    def _generate_stream(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with streaming"""
        print("💭 Generating response (streaming)...")
        
        system_message, conversation = self._convert_messages_to_gemini_format(messages)
        
        # Create chat with history
        self.chat = self.model.start_chat(history=conversation[:-1] if len(conversation) > 1 else [])
        
        # Get last user message
        last_message = conversation[-1]['parts'][0] if conversation else ""
        
        # Add system message if present
        if system_message:
            last_message = f"{system_message}\n\n{last_message}"
        
        full_response = ""
        
        try:
            # Stream response
            response = self.chat.send_message(last_message, stream=True)
            
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    print(chunk.text, end="", flush=True)
            
            print()  # Newline after streaming
        except Exception as e:
            print(f"\n⚠️  Streaming error: {e}")
            # Fallback to non-streaming
            response = self.chat.send_message(last_message, stream=False)
            full_response = response.text
        
        return full_response
    
    def _generate_sync(self, messages: List[Dict[str, str]]) -> str:
        """Generate response without streaming"""
        print("💭 Generating response...")
        
        system_message, conversation = self._convert_messages_to_gemini_format(messages)
        
        # Create chat with history
        self.chat = self.model.start_chat(history=conversation[:-1] if len(conversation) > 1 else [])
        
        # Get last user message
        last_message = conversation[-1]['parts'][0] if conversation else ""
        
        # Add system message if present
        if system_message:
            last_message = f"{system_message}\n\n{last_message}"
        
        try:
            response = self.chat.send_message(last_message)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini generation failed: {e}")
    
    def generate_with_context(
        self, 
        question: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response with optional context
        
        Args:
            question: User's question
            context: Optional context/background information
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful, knowledgeable tutor. Provide clear, educational explanations."
            })
        
        # Add context if provided
        if context:
            messages.append({
                "role": "user",
                "content": f"Context: {context}"
            })
        
        # Add question
        messages.append({
            "role": "user",
            "content": question
        })
        
        return self.generate(messages, stream=False)
    
    def chat_conversation(
        self, 
        question: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Chat with conversation history
        
        Args:
            question: Current question
            conversation_history: Previous messages
            
        Returns:
            Response
        """
        messages = conversation_history or []
        
        # Add current question
        messages.append({
            "role": "user",
            "content": question
        })
        
        response = self.generate(messages, stream=False)
        
        return response


class TutorGeminiEngine(GeminiEngine):
    """
    Specialized Gemini engine for tutoring
    Extends base Gemini with tutor-specific features
    """
    
    def __init__(self, subject: Optional[str] = None, **kwargs):
        """
        Initialize tutor Gemini
        
        Args:
            subject: Subject area (e.g., "Mathematics", "Physics")
            **kwargs: Additional arguments passed to GeminiEngine
        """
        super().__init__(**kwargs)
        self.subject = subject
        self.conversation_history = []
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on subject"""
        base_prompt = """You are an expert tutor. Your role is to:
- Explain concepts clearly and simply
- Provide examples when helpful
- Break down complex topics into understandable parts
- Encourage students when they struggle
- Answer questions thoroughly but concisely
- Use analogies and real-world examples"""
        
        if self.subject:
            base_prompt = f"You are an expert {self.subject} tutor. " + base_prompt
        
        return base_prompt
    
    def ask(self, question: str, reset_history: bool = False) -> str:
        """
        Ask tutor a question
        
        Args:
            question: Student's question
            reset_history: Whether to clear conversation history
            
        Returns:
            Tutor's response
        """
        if reset_history:
            self.conversation_history = []
            self.chat = None
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": question})
        
        # Generate response
        response = self.generate(messages, stream=False)
        
        # Update history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def set_subject(self, subject: str):
        """Change subject area"""
        self.subject = subject
        self.system_prompt = self._build_system_prompt()
        print(f"📚 Subject set to: {subject}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.chat = None
        print("🔄 Conversation history cleared")


# ============================================================
# Example Usage & Testing
# ============================================================

def test_gemini_engine():
    """Test Gemini engine"""
    print("\n" + "="*60)
    print("Testing Google Gemini Engine")
    print("="*60)
    
    # Initialize
    engine = GeminiEngine()
    
    # Test 1: Simple question
    print("\n📝 Test 1: Simple Question")
    print("-" * 60)
    
    messages = [
        {"role": "user", "content": "What is machine learning in simple terms?"}
    ]
    
    response = engine.generate(messages, stream=False)
    
    print(f"\n✓ Response received ({len(response)} chars)")
    print(f"Preview: {response[:200]}...")
    
    # Test 2: With context
    print("\n📝 Test 2: Question with Context")
    print("-" * 60)
    
    response = engine.generate_with_context(
        question="Explain gradient descent",
        context="We are learning about neural networks and optimization algorithms",
        system_prompt="You are a machine learning tutor. Explain concepts clearly with examples."
    )
    
    print(f"\n✓ Response received ({len(response)} chars)")
    print(f"Preview: {response[:200]}...")


def test_tutor_engine():
    """Test tutor-specific engine"""
    print("\n" + "="*60)
    print("Testing Tutor Gemini Engine")
    print("="*60)
    
    # Initialize tutor
    tutor = TutorGeminiEngine(subject="Mathematics")
    
    # Ask questions
    questions = [
        "What is a derivative?",
        "Can you show me an example?",
        "How do I find the derivative of x^2?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {q}")
        print("-" * 60)
        
        response = tutor.ask(q)
        print(f"\n✓ Response received ({len(response)} chars)")
        print(f"Preview: {response[:200]}...")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment")
        print("\n💡 Setup:")
        print("   1. Get API key from https://aistudio.google.com/app/apikey")
        print("   2. Set environment variable:")
        print("      export GOOGLE_API_KEY='your-key-here'")
        print("   3. Or add to .env file:")
        print("      GOOGLE_API_KEY=your-key-here")
    else:
        test_gemini_engine()
        test_tutor_engine()
