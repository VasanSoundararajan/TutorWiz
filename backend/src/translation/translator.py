"""
Multilingual Translation using mBART
Enables tutoring in multiple languages
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


class Translator:
    """
    Multilingual translation using mBART-50
    Supports 50+ languages for inclusive education
    """
    
    # Language codes for mBART-50
    SUPPORTED_LANGUAGES = {
        'en': 'en_XX',  # English
        'hi': 'hi_IN',  # Hindi
        'ta': 'ta_IN',  # Tamil
        'te': 'te_IN',  # Telugu
        'bn': 'bn_IN',  # Bengali
        'mr': 'mr_IN',  # Marathi
        'es': 'es_XX',  # Spanish
        'fr': 'fr_XX',  # French
        'de': 'de_DE',  # German
        'zh': 'zh_CN',  # Chinese
        'ja': 'ja_XX',  # Japanese
        'ko': 'ko_KR',  # Korean
        'ar': 'ar_AR',  # Arabic
        'ru': 'ru_RU',  # Russian
    }
    
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt", device: str = None):
        """
        Initialize translator
        
        Args:
            model_name: Hugging Face model name
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🌐 Loading translation model: {model_name}")
        print(f"   Device: {self.device}")
        
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        
        print(f"✓ Translator ready ({len(self.SUPPORTED_LANGUAGES)} languages)")
    
    def translate(
        self,
        text: str,
        source_lang: str = 'en',
        target_lang: str = 'hi',
        max_length: int = 512
    ) -> str:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code ('en', 'hi', 'ta', etc.)
            target_lang: Target language code
            max_length: Maximum length of translation
            
        Returns:
            Translated text
        """
        # Map language codes
        src_lang_code = self.SUPPORTED_LANGUAGES.get(source_lang, 'en_XX')
        tgt_lang_code = self.SUPPORTED_LANGUAGES.get(target_lang, 'hi_IN')
        
        print(f"🔄 Translating: {source_lang} → {target_lang}")
        
        # Set source language
        self.tokenizer.src_lang = src_lang_code
        
        # Tokenize
        encoded = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate translation
        forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_code]
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return translation.strip()
    
    def batch_translate(
        self,
        texts: List[str],
        source_lang: str = 'en',
        target_lang: str = 'hi'
    ) -> List[str]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        translations = []
        
        for text in texts:
            translation = self.translate(text, source_lang, target_lang)
            translations.append(translation)
        
        return translations
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text (simple heuristic)
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # This is a placeholder - for production, use a proper language detector
        # like langdetect or fasttext
        
        # Simple heuristic based on Unicode ranges
        for char in text:
            # Devanagari script (Hindi, Marathi, etc.)
            if '\u0900' <= char <= '\u097F':
                return 'hi'
            # Tamil script
            elif '\u0B80' <= char <= '\u0BFF':
                return 'ta'
            # Telugu script
            elif '\u0C00' <= char <= '\u0C7F':
                return 'te'
            # Bengali script
            elif '\u0980' <= char <= '\u09FF':
                return 'bn'
        
        # Default to English
        return 'en'
    
    def is_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.SUPPORTED_LANGUAGES
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.SUPPORTED_LANGUAGES.keys())


class SmartTranslator(Translator):
    """
    Enhanced translator with auto-detection and caching
    """
    
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt", device: str = None):
        super().__init__(model_name, device)
        self.cache = {}  # Translation cache
    
    def translate_auto(self, text: str, target_lang: str = 'hi') -> str:
        """
        Translate with automatic source language detection
        
        Args:
            text: Text to translate
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        # Detect source language
        source_lang = self.detect_language(text)
        
        # Check cache
        cache_key = f"{text}|{source_lang}|{target_lang}"
        if cache_key in self.cache:
            print("✓ Using cached translation")
            return self.cache[cache_key]
        
        # Translate
        translation = self.translate(text, source_lang, target_lang)
        
        # Cache result
        self.cache[cache_key] = translation
        
        return translation


# ============================================================
# Example Usage
# ============================================================

def example_usage():
    """Example usage of translator"""
    
    translator = Translator()
    
    # English to Hindi
    text_en = "Machine learning is a subset of artificial intelligence."
    text_hi = translator.translate(text_en, 'en', 'hi')
    print(f"EN: {text_en}")
    print(f"HI: {text_hi}")
    
    # English to Tamil
    text_ta = translator.translate(text_en, 'en', 'ta')
    print(f"TA: {text_ta}")


if __name__ == "__main__":
    example_usage()
