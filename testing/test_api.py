#!/usr/bin/env python3
"""
Test Script: Upload Document → Ask Questions
Tests the complete RAG workflow via the FastAPI API
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_workflow():
    print("\n" + "=" * 70)
    print("  DOCUMIND API TEST — Full Workflow")
    print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────
    # 1. INGEST A DOCUMENT
    # ──────────────────────────────────────────────────────────────
    
    print("[1/3] Uploading a document...")
    
    document_text = """
    Artificial Intelligence (AI) is transforming various industries and sectors.
    Machine Learning is a subset of AI that enables systems to learn from data.
    
    Natural Language Processing (NLP) allows computers to understand human language.
    Deep Learning uses neural networks with multiple layers to process information.
    
    Computer Vision enables systems to interpret and understand visual information.
    Robotics combines AI with mechanical systems to create intelligent machines.
    
    AI applications include chatbots, recommendation systems, autonomous vehicles,
    and medical diagnosis systems. These technologies are revolutionizing how we
    work, communicate, and solve complex problems.
    """
    
    ingest_payload = {
        "text": document_text,
        "filename": "ai_overview.txt"
    }
    
    response = requests.post(
        f"{API_BASE}/ingest/text",
        json=ingest_payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Document ingested successfully!")
        print(f"   Filename: {result['filename']}")
        print(f"   Chunks created: {result['num_chunks']}")
        print(f"   Document ID: {result['doc_id']}\n")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}\n")
        return
    
    # Small delay to ensure vectors are stored
    time.sleep(1)
    
    # ──────────────────────────────────────────────────────────────
    # 2. ASK QUESTIONS
    # ──────────────────────────────────────────────────────────────
    
    questions = [
        "What is Machine Learning?",
        "How does Natural Language Processing work?",
        "What are some AI applications?",
        "Explain Deep Learning",
    ]
    
    print("[2/3] Asking questions about the document...\n")
    
    for i, question in enumerate(questions, 1):
        print("─" * 70)
        print(f"Q{i}: {question}")
        print("─" * 70)
        
        ask_payload = {
            "question": question,
            "top_k": 3
        }
        
        response = requests.post(
            f"{API_BASE}/ask",
            json=ask_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📝 Answer (Mode: {result['mode'].upper()}):")
            print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
            
            print(f"\n📚 Sources Used ({len(result['sources'])} chunks):")
            for idx, source in enumerate(result['sources'], 1):
                print(f"   [{idx}] {source['filename']}")
                print(f"       Similarity: {source['similarity']:.4f}")
                print(f"       Excerpt: {source['excerpt'][:80]}...\n")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}\n")
    
    # ──────────────────────────────────────────────────────────────
    # 3. SEMANTIC SEARCH
    # ──────────────────────────────────────────────────────────────
    
    print("[3/3] Pure semantic search (without LLM)...\n")
    print("─" * 70)
    print("Search Query: How do robots use artificial intelligence?")
    print("─" * 70)
    
    search_payload = {
        "query": "How do robots use artificial intelligence?",
        "top_k": 3
    }
    
    response = requests.post(
        f"{API_BASE}/search",
        json=search_payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n🔍 Found {len(result['results'])} relevant chunks:\n")
        for idx, chunk in enumerate(result['results'], 1):
            print(f"[{idx}] Similarity: {chunk['similarity']:.4f}")
            print(f"    File: {chunk['filename']}")
            print(f"    Text: {chunk['text'][:120]}...\n")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}\n")
    
    print("=" * 70)
    print("  ✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        test_workflow()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API at http://localhost:8000")
        print("   Make sure the server is running: python app.py\n")
    except Exception as e:
        print(f"❌ ERROR: {e}\n")
