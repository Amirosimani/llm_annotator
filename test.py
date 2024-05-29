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
from config import PALM_CONFIG


async def palm(prompt:str, palm_config=PALM_CONFIG) -> List:

    from vertexai.language_models import TextGenerationModel

    vertexai.init(project=palm_config["project_config"]["project"], 
                    location=palm_config["project_config"]["location"])


    rate_limiter = Limiter(palm_config["project_config"]["qpm"]/60) # Limit to 300 requests per 60 second
    model = TextGenerationModel.from_pretrained(palm_config["model"])
    await rate_limiter.wait()
    try:
        responses = model.predcit(
            prompt,
            **palm_config["generation_config"]
            )
    except Exception as e:
        print(f"Error in __palm: {e}") 
        raise
    return(responses.text)

# Define an async function to call the palm method
async def get_palm_response(prompt):
    responses = await palm(prompt)
    print(responses)

if __name__ == "__main__":

    asyncio.run(get_palm_response("Your prompt here"))
