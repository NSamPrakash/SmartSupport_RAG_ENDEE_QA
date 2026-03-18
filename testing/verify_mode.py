#!/usr/bin/env python3
"""
Verify API mode is GENERATIVE
"""

import os
from src.qa_pipeline import QAPipeline
from src.rag_engine import RAGEngine

print("\n" + "="*70)
print("  SmartSupport - API Mode Configuration Check")
print("="*70 + "\n")

# Check environment
api_key = os.getenv("GEMINI_API_KEY", "")
api_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

print("Environment Configuration:")
print(f"  * API Key configured: {('YES [ACTIVE]' if api_key else 'NO [NOT CONFIGURED]')}")
print(f"  * Model: {api_model}")

# Initialize pipeline
print("\nInitializing QA Pipeline...")
qa = QAPipeline()

print(f"\nMode Status:")
print(f"  * GENERATIVE mode enabled: {('YES' if qa.use_llm else 'NO')}")
print(f"  * Will use Gemini API: {('YES - for related queries' if qa.use_llm else 'NO - extractive only')}")

print("\n" + "="*70)
print("\nBehavior:")
print("  1. Related queries (similarity > 50%)")
print("     -> Uses GENERATIVE mode (Gemini API)")
print("  2. Unrelated queries (similarity < 50%)")
print("     -> Returns NOT_RELATED message (skips LLM to save quota)")
print("  3. If API fails")
print("     -> Falls back to EXTRACTIVE mode")
print("\n" + "="*70 + "\n")
