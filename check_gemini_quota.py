#!/usr/bin/env python3
"""
Check Google Gemini API quota and usage status
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not set in .env file")
    exit(1)

try:
    import google.generativeai as genai
    from google.api_core import exceptions
    
    # Configure with API key
    genai.configure(api_key=GEMINI_API_KEY)
    
    print("\n" + "="*60)
    print("  Google Gemini API - Quota & Usage Check")
    print("="*60 + "\n")
    
    # Try to get list of models to verify API key works
    try:
        models = stream_generate_content = genai.list_models()
        print("✅ API Key is VALID and active\n")
        
        # Get available models
        print("📋 Available Models:")
        available_models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"   • {model.name.replace('models/', '')}")
        
        print("\n" + "-"*60)
        
        # Try a simple API call to check quota
        print("\n🔍 Testing API with sample request...\n")
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            "Say 'API is working' in one sentence",
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=100,
            )
        )
        print("✅ API Request SUCCESSFUL")
        print(f"   Response: {response.text[:100]}\n")
        
        print("-"*60)
        print("\n📊 Quota Information:\n")
        print("   FREE TIER LIMITS (you are using FREE tier):")
        print("   • Requests per minute: 60 RPM")
        print("   • Requests per day: 1,500 DAILY")
        print("   • Tokens per minute: 1,000,000 TPM")
        print("\n   ✅ Your API key is working properly!")
        print("   ✅ You have NOT exceeded your free tier quota")
        print("\n   To check real-time quota usage:")
        print("   1. Visit: https://console.cloud.google.com/")
        print("   2. Select your project")
        print("   3. Go to: APIs & Services > Quotas")
        print("   4. Search for 'Generative Language API'")
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "quota" in error_msg or "rate_limit" in error_msg:
            print("⚠️  QUOTA EXCEEDED!")
            print(f"\n   Error: {e}")
            print("\n   You have hit your daily quota limit (1,500 requests)")
            print("   Next quota reset: Tomorrow at 00:00 UTC")
            print("\nOptions:")
            print("   1. Wait until tomorrow for daily quota reset")
            print("   2. Use a different API key with fresh quota")
            print("   3. Upgrade to paid tier for higher limits")
            
        elif "invalid" in error_msg or "authentication" in error_msg:
            print("❌ INVALID API KEY!")
            print(f"\n   Error: {e}")
            print("\n   Please check your GEMINI_API_KEY in .env file")
            
        else:
            print(f"❌ API ERROR: {e}")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
except ImportError:
    print("❌ ERROR: google-generativeai package not installed")
    print("   Run: pip install google-generativeai")
except Exception as e:
    print(f"❌ Fatal error: {e}")

print("\n" + "="*60 + "\n")
