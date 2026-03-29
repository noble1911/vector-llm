"""Quick test: does the model use tools correctly?"""
import asyncio
import sys
import yaml
import httpx

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:9b"

sys.path.insert(0, ".")

with open("config/personality.yaml") as f:
    config = yaml.safe_load(f)

prompt = config["personality"]["system_prompt"]
prompt += "\n[Current vision] objects: mouse, laptop | motion: none"

from src.brain import CHAT_TOOLS, ACTION_TOOLS


async def test(user_msg):
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post("http://localhost:11434/api/chat", json={
            "model": MODEL,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            "tools": ACTION_TOOLS,
            "options": {"num_predict": 120},
        })
        data = resp.json()
        msg = data["message"]
        tc = msg.get("tool_calls", [])
        duration = data.get("total_duration", 0) / 1e9
        print(f"Q: {user_msg}")
        print(f"  speech: {msg.get('content', '')[:80]}")
        if tc:
            for t in tc:
                args = t["function"].get("arguments", {})
                print(f"  tool: {t['function']['name']}({args})")
        else:
            print(f"  tools: none")
        print(f"  time: {duration:.1f}s")
        print()


async def main():
    await test("What can you see?")
    await test("My name is Ron.")
    await test("My favourite colour is blue, remember that.")
    await test("Move forward please.")
    await test("Turn left 90 degrees.")
    await test("Go back to your charger.")
    await test("What's my name?")
    await test("What's my favourite colour?")

asyncio.run(main())
