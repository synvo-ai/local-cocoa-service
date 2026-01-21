import asyncio
import httpx

URL = "http://127.0.0.1:8007"

async def test_complete():
    print("Testing /completion...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{URL}/completion",
                json={
                    "prompt": "Hello, who are you?",
                    "max_tokens": 50
                }
            )
            print(f"Status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error: {resp.text}")
            else:
                print(f"Response: {resp.json()}")
        except Exception as e:
            print(f"Exception: {e}")

async def test_chat():
    print("\nTesting /v1/chat/completions...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{URL}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello, who are you?"}],
                    "max_tokens": 50
                }
            )
            print(f"Status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error: {resp.text}")
            else:
                print(f"Response: {resp.json()}")
        except Exception as e:
            print(f"Exception: {e}")

async def main():
    await test_complete()
    await test_chat()

if __name__ == "__main__":
    asyncio.run(main())
