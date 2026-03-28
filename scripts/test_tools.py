"""Quick test: does the model use tools correctly?"""
import asyncio
import sys
import yaml
import httpx

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"

sys.path.insert(0, ".")

with open("config/personality.yaml") as f:
    config = yaml.safe_load(f)

prompt = config["personality"]["system_prompt"]
prompt += "\n[Current vision] objects: mouse, laptop | motion: none"

from src.brain import TOOLS


async def test(user_msg):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post("http://localhost:11434/api/chat", json={
            "model": MODEL,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            "tools": TOOLS,
            "options": {"num_predict": 120},
        })
        data = resp.json()
        msg = data["message"]
        tc = msg.get("tool_calls", [])
        print(f"Q: {user_msg}")
        print(f"  speech: {msg.get('content', '')[:80]}")
        print(f"  tools: {[t['function']['name'] for t in tc] if tc else 'none'}")
        print()


async def main():
    await test("What can you see?")
    await test("My name is Ron.")
    await test("Move forward please.")
    await test("Turn left 90 degrees.")
    await test("Go back to your charger.")
    await test("Make your eyes green.")

asyncio.run(main())
