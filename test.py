import re
import logging
import asyncio
import numpy as np
import scipy as sp
from collections import Counter
from itertools import combinations
from asynciolimiter import Limiter
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Optional

import vertexai
from config import CLAUDE_CONFIG

# async def claude(prompt:str, claude_config: dict) -> List:

#     from anthropic import AnthropicVertex
#     print(claude_config)

#     client = AnthropicVertex(region=claude_config["project_config"]["location"], 
#                                 project_id=claude_config["project_config"]["project"])


#     rate_limiter = Limiter(claude_config["project_config"]["qpm"]/60) # Limit to 60 requests per 60 second

#     await rate_limiter.wait()
#     try:
#         responses = client.messages.create(
#             max_tokens=1024,
#             messages=[
#                 {"role": "user",
#                 "content": prompt,
#                 }
#                 ],
#                 model=claude_config["model"],
#                 )
#     except Exception as e:
#         raise
#     return(responses.content)


def claude(prompt:str):

    from anthropic import AnthropicVertex

    MODEL = "claude-3-haiku@20240307"

    client = AnthropicVertex(region="us-central1", project_id="amir-genai-bb")

    message = client.messages.create(
        max_tokengits=1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=MODEL,
    )
    print(message.model_dump_json(indent=2))

# # Define an async function to call the palm method
# async def get_palm_response(prompt, claude_config):
#     responses = await claude(prompt, claude_config)
#     print(responses)

# if __name__ == "__main__":

    # asyncio.run(get_palm_response("Your prompt here", CLAUDE_CONFIG))
claude("give me a banana bread recipe")
