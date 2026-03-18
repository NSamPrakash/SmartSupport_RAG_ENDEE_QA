#!/usr/bin/env python3
"""
Test Script: Relevance Filter
Tests that the RAG system only answers questions related to uploaded documents
and refuses to answer unrelated queries.
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_relevance():
    print("\n" + "=" * 80)
    print("  DOCUMIND RELEVANCE TEST — Reject Out-of-Scope Questions")
    print("=" * 80 + "\n")

    # ──────────────────────────────────────────────────────────────
    # 1. UPLOAD ONLY TECH/AI DOCUMENTS
    # ──────────────────────────────────────────────────────────────
    
    print("[1/3] Uploading specialized tech documents only...\n")
    
    tech_document = """
    Quantum Computing Overview:
    Quantum computers use quantum bits (qubits) instead of classical bits.
    Qubits can exist in superposition, allowing them to process multiple 
    states simultaneously. This enables quantum computers to solve certain 
    problems exponentially faster than classical computers.
    
    Key quantum computing concepts:
    - Superposition: qubits can be 0, 1, or both simultaneously
    - Entanglement: qubits can be correlated in non-classical ways
    - Quantum interference: amplifying right answers, canceling wrong ones
    
    Applications include cryptography, drug discovery, and optimization.
    Companies like IBM, Google, and Amazon are developing quantum systems.
    """
    
    ingest_payload = {
        "text": tech_document,
        "filename": "quantum_computing.txt"
    }
    
    response = requests.post(
        f"{API_BASE}/ingest/text",
        json=ingest_payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Document ingested: {result['filename']}")
        print(f"   Chunks: {result['num_chunks']}\n")
    else:
        print(f"❌ Error: {response.text}\n")
        return
    
    time.sleep(1)
    
    # ──────────────────────────────────────────────────────────────
    # 2. TEST RELEVANT QUESTIONS (Should get good answers)
    # ──────────────────────────────────────────────────────────────
    
    print("[2/3] Testing RELEVANT queries (should answer):\n")
    
    relevant_questions = [
        "What is quantum computing?",
        "Explain quantum superposition",
        "What are applications of quantum computers?",
    ]
    
    for i, question in enumerate(relevant_questions, 1):
        print("─" * 80)
        print(f"✓ Q{i}: {question}")
        print("─" * 80)
        
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
            print(f"Mode: {result['mode'].upper()}")
            
            if result['sources']:
                top_sim = result['sources'][0]['similarity']
                print(f"Top relevance score: {top_sim:.4f}")
                print(f"Answer: {result['answer'][:200]}...\n")
            else:
                print(f"⚠️  No sources found\n")
        else:
            print(f"❌ Error: {response.status_code}\n")
    
    # ──────────────────────────────────────────────────────────────
    # 3. TEST IRRELEVANT QUESTIONS (Should refuse/return low scores)
    # ──────────────────────────────────────────────────────────────
    
    print("\n[3/3] Testing IRRELEVANT queries (should refuse/low relevance):\n")
    
    irrelevant_questions = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "What is the best football team?",
        "Tell me about ancient Egyptian history",
        "How to fix a car engine?",
    ]
    
    for i, question in enumerate(irrelevant_questions, 1):
        print("─" * 80)
        print(f"✗ Q{i}: {question}")
        print("─" * 80)
        
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
            print(f"Mode: {result['mode'].upper()}")
            
            if result['sources']:
                top_sim = result['sources'][0]['similarity']
                print(f"⚠️  Top relevance score: {top_sim:.4f} (LOW = Good filtering)")
                
                if top_sim < 0.3:
                    print("✓ PASS: Low similarity - correctly rejected")
                else:
                    print("✗ FAIL: High similarity - might be false positive")
                
                print(f"Answer: {result['answer'][:150]}...\n")
            else:
                print("✓ PASS: No sources found - correctly rejected\n")
        else:
            print(f"Error: {response.status_code}\n")
    
    print("=" * 80)
    print("  RELEVANCE TEST COMPLETE")
    print("  ✓ Relevant questions should have high similarity (>0.5)")
    print("  ✓ Irrelevant questions should have low similarity (<0.3)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        test_relevance()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API at http://localhost:8000")
        print("   Make sure the server is running: python app.py\n")
    except Exception as e:
        print(f"❌ ERROR: {e}\n")
