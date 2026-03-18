#!/usr/bin/env python3
"""
Test similarity threshold feature - queries related and unrelated to ML documents
"""

from src.rag_engine import RAGEngine
from src.qa_pipeline import QAPipeline

# Test ML-related queries
ML_RELATED_QUERIES = [
    "What is machine learning?",
    "Explain supervised learning",
    "Tell me about neural networks",
    "How does NLP work?",
]

# Test UNRELATED queries (should trigger "not_related" mode)
UNRELATED_QUERIES = [
    "What is the capital of France?",
    "How do I cook pasta?",
    "Tell me about ancient Rome",
    "What is the weather today?",
    "Explain quantum physics",
]

def test_similarity_threshold():
    print("\n" + "="*70)
    print("  SmartSupport - Similarity Threshold Test")
    print("="*70)

    # Initialize
    print("\n[1/3] Initialising RAG engine...")
    rag = RAGEngine()
    qa = QAPipeline(rag_engine=rag)
    print("  ✓ Connected\n")

    # Test related queries
    print("[2/3] Testing ML-RELATED queries (should show sources)...\n")
    print("─" * 70)
    for q in ML_RELATED_QUERIES:
        result = qa.ask(q, top_k=3)
        max_sim = max([s['similarity'] for s in result['sources']], default=0)
        status = "✅ RELATED" if max_sim >= 0.5 else "❌ NOT RELATED"
        print(f"\nQ: {q}")
        print(f"   Mode: {result['mode'].upper()}")
        print(f"   Max Similarity: {max_sim:.4f} ({max_sim*100:.1f}%)")
        print(f"   Status: {status}")
        print(f"   Answer: {result['answer'][:80]}...")
        if result['sources']:
            print(f"   Sources: {len(result['sources'])} found")

    # Test unrelated queries
    print("\n" + "─" * 70)
    print("\n[3/3] Testing UNRELATED queries (should show 'NOT RELATED')...\n")
    print("─" * 70)
    for q in UNRELATED_QUERIES:
        result = qa.ask(q, top_k=3)
        max_sim = max([s['similarity'] for s in result['sources']], default=0)
        status = "✅ CORRECT" if result['mode'] == 'not_related' else "❌ MISLABELED"
        print(f"\nQ: {q}")
        print(f"   Mode: {result['mode'].upper()}")
        print(f"   Max Similarity: {max_sim:.4f} ({max_sim*100:.1f}%)")
        print(f"   Status: {status}")
        print(f"   Answer: {result['answer'][:60]}...")

    print("\n" + "="*70)
    print("  Test Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_similarity_threshold()
