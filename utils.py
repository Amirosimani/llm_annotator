import re
import asyncio
from asynciolimiter import Limiter

import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

QPM = 5
PROJECT = "amir-genai-bb"
LOCATION = "us-central1"

rate_limiter = Limiter(QPM/60) # Limit to 5 requests per 60 second

def _extract_binary_values(response_text):
    """Extracts 0 or 1 values from a list of strings, prioritizing single digits.
    Args:
        string_list: A list of strings potentially containing 0s or 1s.
    Returns:
        A list of integers (0 or 1) extracted from the strings.
    """

    pattern = r"\b[01]\b"  # Matches standalone 0 or 1

    match = re.search(pattern, response_text)
    if match:
        return int(match.group())  # Convert to integer
    else:
        print("Response format is not correct")
        None


async def gemini(prompt):
    vertexai.init(project=PROJECT, location=LOCATION)

    generation_config = {
        "max_output_tokens": 2048,
        "temperature": 0.1,
        "top_p": 1,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    model = GenerativeModel("gemini-1.0-pro-002",
                           system_instruction=[
                             "You are a helpful data labeler.",
                             "Your mission is to label the input data based on the instruction you will receive.",
                             ],
                        )
    await rate_limiter.wait()

    responses = model.generate_content(
      [prompt],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=False,
  )
    return(_extract_binary_values(responses.text))


async def main(prompts, model):

    if model == "gemini":
        tasks = []
        for url in prompts:
            tasks.append(asyncio.create_task(gemini(url)))
        
        results = await asyncio.gather(*tasks) # the order of result values is preserved, but the execution order is not. https://docs.python.org/3/library/asyncio-task.html#running-tasks-concurrently
        print(results)
        return results
    else:
        print("Wrong model")


if __name__ == "__main__":
    import time
    s = time.perf_counter()
    prompts = ["where is paris?", "whare is tehran"]
    asyncio.run(main(prompts))
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")


