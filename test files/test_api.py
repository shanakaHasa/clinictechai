"""
Simple API test script
"""

import asyncio
import json
import httpx

async def test_health():
    """Test health endpoint"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get("http://localhost:8000/health")
            print("=== HEALTH CHECK ===")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            print()
        except Exception as e:
            print(f"Health check failed: {e}")
            print()

async def test_query():
    """Test query endpoint"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            payload = {"query": "What is the clinical finding?"}
            response = await client.post(
                "http://localhost:8000/query",
                json=payload
            )
            print("=== QUERY TEST ===")
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            print()
        except Exception as e:
            print(f"Query test failed: {e}")
            print()

async def main():
    print("Testing ClinicTech AI API")
    print("=" * 50)
    await test_health()
    await test_query()

if __name__ == "__main__":
    asyncio.run(main())
