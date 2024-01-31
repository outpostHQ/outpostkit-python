import json
from typing import Any, Dict

import httpx


async def stream_vllm_responses(url: str, data: Dict[str, Any]):
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data, timeout=None)
        response.raise_for_status()

        async for line in response.aiter_lines():
            # Each line here is a separate message in the stream.
            if line:
                message = json.loads(line)
                print(message)
