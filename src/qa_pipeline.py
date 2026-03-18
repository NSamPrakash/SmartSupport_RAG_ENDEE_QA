"""
SmartSupport QA Pipeline
Wraps the RAG engine with an LLM to provide grounded answers.
Supports Google Gemini and a lightweight local fallback (no-LLM extractive mode).
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

from src.rag_engine import RAGEngine, SearchResult

# Load environment variables from .env file (force reload)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


# ──────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────

def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call the Google Gemini API."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000,
            )
        )
        logger.info(f"✅ Gemini API response successful (model: {model_name})")
        return response.text.strip()
    except Exception as e:
        logger.warning(f"⚠️ Gemini API failed, falling back to extractive mode: {e}")
        # Fall back to extractive mode
        return None


def _extractive_answer(query: str, results: list[SearchResult]) -> str:
    """
    Lightweight fallback: returns the top matching chunk as the answer.
    Useful when no LLM API key is configured.
    """
    if not results:
        return "No relevant information found in the knowledge base."
    top = results[0]
    return (
        f"[Extractive answer from '{top.filename}' — similarity {top.similarity:.3f}]\n\n"
        f"{top.text}"
    )


# ──────────────────────────────────────────────
# QA Pipeline
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are SmartSupport, a precise and helpful support Q&A assistant.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information, say so clearly.
Cite the source filename when possible. Be concise yet thorough."""


class QAPipeline:
    """
    High-level QA pipeline:
      1. Retrieve relevant chunks via Endee (RAGEngine)
      2. Build a context string
      3. Call an LLM (or fallback to extractive mode)
    """

    def __init__(self, rag_engine: Optional[RAGEngine] = None):
        self.rag = rag_engine or RAGEngine()
        self.use_llm = bool(os.getenv("GEMINI_API_KEY", ""))
        if self.use_llm:
            logger.info("✅ GENERATIVE mode enabled - Using Gemini API for intelligent responses")
        else:
            logger.warning("⚠️  EXTRACTIVE mode only - No API key configured, using document extraction fallback")

    def ask(self, question: str, top_k: int = 5, similarity_threshold: float = 0.5) -> dict:
        """
        Answer a question using the RAG pipeline.

        Returns a dict with:
          - question:  original question
          - answer:    generated (or extractive) answer
          - sources:   list of source chunk metadata used
          - mode:      'generative' | 'extractive' | 'not_related'
        """
        context, results = self.rag.build_context(question, top_k=top_k)

        if not results:
            return {
                "question": question,
                "answer":   "I don't have relevant information about this topic in my knowledge base. Could you try rephrasing your question?",
                "sources":  [],
                "mode":     "not_related",
            }
        
        # Check if all results have low similarity (below 50% threshold)
        max_similarity = max(r.similarity for r in results) if results else 0
        if max_similarity < similarity_threshold:
            return {
                "question": question,
                "answer":   "I couldn't find relevant information to answer your question. Please try asking about topics covered in the uploaded documents or rephrase your query.",
                "sources":  [],
                "mode":     "not_related",
            }

        if self.use_llm:
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
            answer = _call_gemini(SYSTEM_PROMPT, user_prompt)
            if answer is None:
                # LLM failed, fall back to extractive mode
                answer = _extractive_answer(question, results)
                mode = "extractive_fallback"
            else:
                mode = "generative"
        else:
            # Extractive mode (no LLM available)
            answer = _extractive_answer(question, results)
            # Add relevance warning if similarity is low even in extractive mode
            if max_similarity < similarity_threshold:
                answer = f"⚠️  Low Relevance Match\n\n{answer}\n\n(Similarity: {max_similarity:.1%} - Below 50% threshold)"
            mode = "extractive"

        sources = [
            {
                "filename":   r.filename,
                "chunk_id":   r.chunk_id,
                "similarity": round(r.similarity, 4),
                "similarity_percent": f"{r.similarity*100:.1f}%",
                "excerpt":    r.text[:200] + ("…" if len(r.text) > 200 else ""),
            }
            for r in results
        ]

        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "mode":     mode,
        }
