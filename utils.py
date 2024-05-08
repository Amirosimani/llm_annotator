import re
import asyncio
from asynciolimiter import Limiter
from typing import Any, Dict, List, Tuple, Optional

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models


gemini_config = {"project_config": {"qpm":5,
                                    "project": "amir-genai-bb",
                                    "location": "us-central1"},
                "generation_config": {"max_output_tokens": 2048,
                                      "temperature": 0.1,
                                      "top_p": 1,
                                      }
}


class Annotate:
    def __init__(self, gemini_config= gemini_config):

        self.qpm = gemini_config["project_config"]["qpm"]
        self.rate_limiter = Limiter(self.qpm/60) # Limit to 5 requests per 60 second
        
        self.project = gemini_config["project_config"]["project"]
        self.location = gemini_config["project_config"]["location"]

        self.gemini_generation_config = gemini_config["generation_config"]

    @staticmethod
    def __extract_binary_values(response_text: str) -> Optional[int]:
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
            return None


    async def __gemini(self, prompt:str) -> List:
        """
        Asynchronously generates labels for datapoints using a Gemini Pro text 
        dataset order is presevered. safety measures are in place.

        Args:
            prompt: The text input to be classified.

        Returns:
            A boolean value representing the model's classification.


        Raises:
            VertexAIError: If there's an issue with the Vertex AI initialization or model call.
            RateLimitExceededError: If the rate limiter indicates excessive API calls.
            Any exceptions raised by the `Annotate.__extract_binary_values` function.
        """

        
        vertexai.init(project=self.project, location=self.location)

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
        await self.rate_limiter.wait()

        responses = model.generate_content(
        [prompt],
        generation_config=self.gemini_generation_config,
        safety_settings=safety_settings,
        stream=False,
    )
        return(Annotate.__extract_binary_values(responses.text))


    async def TextClassification(self, prompts: List[str], model: str) ->  List[Any]:
        """
        Performs text classification annotation using Gemini.

        Args:
            prompts: A list of text prompts to classify.
            model: The name of the model to use for classification ("gemini" is currently supported).

        Returns:
            A list of classification results (the format depends on the model used).

        Raises:
            ValueError: If an unsupported model is specified.
        """

        if model == "gemini":
            tasks = []
            for url in prompts:
                tasks.append(asyncio.create_task(self.__gemini(url)))
            
            results = await asyncio.gather(*tasks) # the order of result values is preserved, but the execution order is not. https://docs.python.org/3/library/asyncio-task.html#running-tasks-concurrently
            print(results)
            return results
        else:
            raise ValueError(f"Unsupported model: {model}")
        

class Evaluate():

    @staticmethod
    def TextClfWithGT(y_true, y_pred):
        clf_metrics = {"accuracy": accuracy_score(y_true, y_pred),
                       "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
                       "confusion_matrix": confusion_matrix(y_true, y_pred)
                       }
        return clf_metrics



if __name__ == "__main__":
    ann = Annotate()
    import time
    s = time.perf_counter()
    prompts = ["where is paris?", "whare is tehran"]
    asyncio.run(ann.TextClassification(prompts, model="gemini"))
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")


