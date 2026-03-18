"""
Generator Module
LLM response generation with streaming support using Groq API.
Implements prompt template design for RAG-based question answering.
"""

import os
from typing import Generator
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Default model - Llama 3 is fast and capable on Groq
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# System prompt template for RAG-based QA
SYSTEM_PROMPT = """You are an expert AI assistant specialized in analyzing and answering questions about legal documents, terms & conditions, privacy policies, and contracts.

Your task is to provide accurate, helpful, and well-structured answers based ONLY on the provided context from the documents.

IMPORTANT RULES:
1. Answer ONLY based on the information provided in the context below.
2. If the context does not contain enough information to fully answer the question, clearly state: "Based on the available documents, I don't have enough information to fully answer this question."
3. Do NOT make up or hallucinate any information not present in the context.
4. When possible, quote or reference specific sections from the context.
5. Provide clear, well-organized responses using bullet points or numbered lists when appropriate.
6. If the question is ambiguous, interpret it in the most reasonable way and note your interpretation.
7. Keep your responses concise but comprehensive."""

# User prompt template
USER_PROMPT_TEMPLATE = """Context from the documents:
---
{context}
---

User Question: {question}

Please provide a detailed and accurate answer based on the context above."""


class Generator:
    """
    LLM response generator using Groq API with streaming support.
    
    Attributes:
        client: Groq API client
        model: Model name to use for generation
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens in the response
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key: str = None,
    ):
        """
        Initialize the generator with Groq API.
        
        Args:
            model: Groq model name
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in the generated response
            api_key: Groq API key (falls back to GROQ_API_KEY env var)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Try to get API key from parameter, then environment variable
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            
        # Try to get API key from Streamlit secrets as fallback
        if not api_key:
            try:
                import streamlit as st
                if "GROQ_API_KEY" in st.secrets:
                    api_key = st.secrets["GROQ_API_KEY"]
            except ImportError:
                pass
            except Exception:
                pass # Streamlit not running or secrets not configured
                
        if not api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in .env file, "
                "Streamlit secrets, or pass api_key parameter."
            )

        self.client = Groq(api_key=api_key)
        print(f"[Generator] Initialized with model '{model}' "
              f"(temp={temperature}, max_tokens={max_tokens})")

    def generate(self, context: str, question: str) -> str:
        """
        Generate a non-streaming response.
        
        Args:
            context: Formatted context from retrieved chunks
            question: User's question
            
        Returns:
            Generated response text
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context, question=question
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )

        return response.choices[0].message.content

    def generate_stream(self, context: str, question: str) -> Generator[str, None, None]:
        """
        Generate a streaming response (token-by-token).
        
        Args:
            context: Formatted context from retrieved chunks
            question: User's question
            
        Yields:
            Individual tokens/chunks of the generated response
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context, question=question
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_model_info(self) -> dict:
        """Return information about the generator configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "Groq",
        }


if __name__ == "__main__":
    # Quick test
    gen = Generator()
    print(f"\nGenerator info: {gen.get_model_info()}")

    # Test with a simple query
    test_context = "The platform's terms of service state that users must be 18 years or older."
    test_question = "What is the minimum age requirement?"

    print("\n--- Streaming Response ---")
    for token in gen.generate_stream(test_context, test_question):
        print(token, end="", flush=True)
    print("\n--- End ---")
