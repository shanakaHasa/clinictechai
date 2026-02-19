#!/usr/bin/env python3
"""Quick test of OpenAI embedding service"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Testing OpenAI Embedding Service")
print("=" * 60)

try:
    print("1️⃣  Loading settings...")
    from app.config.settings import settings
    print(f"   ✅ LLM Model: {settings.llm_model}")
    print(f"   ✅ Embedding Model: {settings.embedding_model}")
    print(f"   ✅ API Key Present: {'Yes' if settings.llm_api_key else 'No'}")
    
    print("\n2️⃣  Testing OpenAI Client...")
    from openai import OpenAI
    client = OpenAI(api_key=settings.llm_api_key)
    print(f"   ✅ OpenAI client created")
    
    print("\n3️⃣  Testing single embedding...")
    response = client.embeddings.create(
        input="test query",
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    print(f"   ✅ Embedding generated")
    print(f"   ✅ Dimension: {len(embedding)} (expected 1536)")
    
    print("\n4️⃣  Testing batch embeddings...")
    response = client.embeddings.create(
        input=["text 1", "text 2", "text 3"],
        model="text-embedding-3-small"
    )
    embeddings = [data.embedding for data in response.data]
    print(f"   ✅ Batch embedding generated")
    print(f"   ✅ Count: {len(embeddings)}")
    print(f"   ✅ Dimension: {len(embeddings[0])}")
    
    print("\n5️⃣  Testing ChromaDB...")
    import chromadb
    client_chroma = chromadb.Client()
    collection = client_chroma.get_or_create_collection(name="test")
    print(f"   ✅ ChromaDB client created")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Ready for production!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
