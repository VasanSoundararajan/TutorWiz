"""
Sarvam-2B Engine - Multilingual Indian Language Model
Uses HuggingFace pipeline for local inference
Location: src/llm/sarvam_engine.py
"""

import os
import torch
from typing import Dict, List, Optional
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

# Try to import transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed")
    print("   Install: pip install transformers torch")


class SarvamEngine:
    """
    Sarvam-2B Engine
    Multilingual language model optimized for Indian languages
    """
    
    def __init__(
        self,
        model_name: str = "sarvamai/sarvam-2b-v0.5",
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Initialize Sarvam engine
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        print(f"🚀 Initializing Sarvam-2B: {model_name}")
        print(f"   Device: {self.device}")
        
        # Load model and tokenizer
        print("   Loading model (this may take a few minutes on first run)...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"✓ Sarvam-2B engine ready")
            
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("   Falling back to smaller model or check model availability")
            raise
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> str:
        """
        Generate response from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response (not supported, will use non-stream)
            
        Returns:
            Generated text response
        """
        # Convert messages to prompt
        prompt = self._format_messages(messages)
        
        print("💭 Generating response...")
        
        # Generate
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        response = outputs[0]['generated_text'].strip()
        return response
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # Add assistant prefix for generation
        prompt += "\n\nAssistant:"
        
        return prompt
    
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


class TutorSarvamEngine(SarvamEngine):
    """
    Specialized Sarvam engine for tutoring
    Extends base Sarvam with tutor-specific features
    """
    
    def __init__(self, subject: Optional[str] = None, language: str = "en", **kwargs):
        """
        Initialize tutor Sarvam
        
        Args:
            subject: Subject area
            language: Language code (en, hi, ta, etc.)
            **kwargs: Additional arguments passed to SarvamEngine
        """
        super().__init__(**kwargs)
        self.subject = subject
        self.language = language
        self.conversation_history = []
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on subject and language"""
        base_prompt = """You are an expert tutor. Your role is to:
- Explain concepts clearly and simply
- Provide examples when helpful
- Break down complex topics into understandable parts
- Encourage students when they struggle
- Answer questions thoroughly but concisely"""
        
        if self.subject:
            base_prompt = f"You are an expert {self.subject} tutor. " + base_prompt
        
        if self.language != "en":
            base_prompt += f"\n\nRespond in {self.language} language when appropriate."
        
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
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": question})
        
        # Generate response
        response = self.generate(messages, stream=False)
        
        # Update history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(self.conversation_history) > 16:  # Smaller for local model
            self.conversation_history = self.conversation_history[-16:]
        
        return response
    
    def set_subject(self, subject: str):
        """Change subject area"""
        self.subject = subject
        self.system_prompt = self._build_system_prompt()
        print(f"📚 Subject set to: {subject}")
    
    def set_language(self, language: str):
        """Change response language"""
        self.language = language
        self.system_prompt = self._build_system_prompt()
        print(f"🌍 Language set to: {language}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🔄 Conversation history cleared")


# ============================================================
# Example Usage & Testing
# ============================================================

def test_sarvam_engine():
    """Test Sarvam engine"""
    print("\n" + "="*60)
    print("Testing Sarvam-2B Engine")
    print("="*60)
    
    # Initialize
    engine = SarvamEngine()
    
    # Test 1: Simple question
    print("\n📝 Test 1: Simple Question")
    print("-" * 60)
    
    messages = [
        {"role": "user", "content": "What is artificial intelligence?"}
    ]
    
    response = engine.generate(messages, stream=False)
    
    print(f"\n✓ Response received ({len(response)} chars)")
    print(f"Response: {response}")
    
    # Test 2: With context
    print("\n📝 Test 2: Question with Context")
    print("-" * 60)
    
    response = engine.generate_with_context(
        question="Explain neural networks",
        context="We are learning about machine learning",
        system_prompt="You are a helpful AI tutor."
    )
    
    print(f"\n✓ Response received ({len(response)} chars)")
    print(f"Preview: {response[:200]}...")


def test_tutor_engine():
    """Test tutor-specific engine"""
    print("\n" + "="*60)
    print("Testing Tutor Sarvam Engine")
    print("="*60)
    
    # Initialize tutor
    tutor = TutorSarvamEngine(subject="Computer Science")
    
    # Ask questions
    questions = [
        "What is an algorithm?",
        "Can you give an example?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {q}")
        print("-" * 60)
        
        response = tutor.ask(q)
        print(f"\n✓ Response: {response}")


if __name__ == "__main__":
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    
    if device == "cpu":
        print("⚠️  Running on CPU. This will be slower.")
        print("   For faster inference, use a GPU.")
    
    test_sarvam_engine()
    # test_tutor_engine()  # Uncomment to test tutor version
