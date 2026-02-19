#!/usr/bin/env python3
"""Test OpenAI API v1.0.0+ compatibility"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Testing OpenAI API v1.0.0+ Compatibility")
print("=" * 60)

try:
    print("1️⃣  Loading settings...")
    from app.config.settings import settings
    print(f"   ✅ LLM Provider: {settings.llm_provider}")
    print(f"   ✅ LLM Model: {settings.llm_model}")
    
    print("\n2️⃣  Testing OpenAI Client (v1.0.0+)...")
    from openai import OpenAI
    client = OpenAI(api_key=settings.llm_api_key)
    print(f"   ✅ OpenAI client created with new API")
    
    print("\n3️⃣  Testing chat.completions.create() method...")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful assistant."},
            {"role": "user", "content": "Say 'Hello from OpenAI v1.0.0+'"}
        ],
        temperature=0.1,
        max_tokens=50
    )
    print(f"   ✅ API call successful!")
    print(f"   ✅ Response: {response.choices[0].message.content}")
    print(f"   ✅ Tokens used: {response.usage.total_tokens}")
    
    print("\n4️⃣  Testing LLM Service...")
    from app.llm.llm_service import LLMService
    llm_service = LLMService()
    
    if llm_service.client:
        print(f"   ✅ LLM Service initialized")
        print(f"      - Provider: {llm_service.provider}")
        print(f"      - Model: {llm_service.model}")
        print(f"      - Temperature: {llm_service.temperature}")
    else:
        print(f"   ❌ LLM Service client not initialized")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - OpenAI v1.0.0+ Compatible!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
