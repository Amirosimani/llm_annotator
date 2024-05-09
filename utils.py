import re
import asyncio
from collections import Counter
from itertools import combinations
from asynciolimiter import Limiter
from typing import Any, Dict, List, Optional


import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

from anthropic import AnthropicVertex


VALID_MODELS = ["gemini", "claude"]


GEMINI_CONFIG = {"project_config": {"qpm":5,
                                    "project": "amir-genai-bb",
                                    "location": "us-central1"},
                "generation_config": {"max_output_tokens": 2048,
                                      "temperature": 0.1,
                                      "top_p": 1,
                                      }
}

CLAUDE_CONFIG = {"project_config": {"qpm":60,  # https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude
                                    "project": "cloud-llm-preview1",
                                    "location": "us-central1"},
                "generation_config": ""

}


class Annotate:
    def __init__(self, gemini_config= GEMINI_CONFIG, claude_config=CLAUDE_CONFIG):

        self.gemini_config = gemini_config
        self.claude_config = claude_config

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

        
        vertexai.init(project=self.gemini_config["project_config"]["project"], 
                      location=self.gemini_config["project_config"]["location"])

        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        rate_limiter = Limiter(self.gemini_config["project_config"]["qpm"]/60) # Limit to 5 requests per 60 second
        model = GenerativeModel("gemini-1.0-pro-002",
                            system_instruction=[
                                "You are a helpful data labeler.",
                                "Your mission is to label the input data based on the instruction you will receive.",
                                ],
                            )
        await rate_limiter.wait()

        responses = model.generate_content(
        [prompt],
        generation_config=self.gemini_config["generation_config"],
        safety_settings=safety_settings,
        stream=False,
    )
        return(Annotate.__extract_binary_values(responses.text))


    async def __claude(self, prompt:str) -> List:
        """
        Asynchronously generates labels for datapoints using Claude Haiku 
        dataset order is presevered.

        Args:
            prompt: The text input to be classified.

        Returns:
            A boolean value representing the model's classification.


        Raises:
            VertexAIError: If there's an issue with the Vertex AI initialization or model call.
            RateLimitExceededError: If the rate limiter indicates excessive API calls.
            Any exceptions raised by the `Annotate.__extract_binary_values` function.
        """
        client = AnthropicVertex(region=self.claude_config["project_config"]["location"], 
                                 project_id=self.claude_config["project_config"]["project"])


        rate_limiter = Limiter(self.claude_config["project_config"]["qpm"]/60) # Limit to 60 requests per 60 second

        await rate_limiter.wait()

        responses = client.messages.create(
            max_tokens=1024,
            messages=[
                {"role": "user",
                 "content": prompt,
                 }
                 ],
                 model="claude-3-haiku@20240307",
                 )
        return(Annotate.__extract_binary_values(responses.content[0].text))



    async def classification(self, prompts: List[str], model: str) ->  List[Any]:
        f"""
        Performs text classification annotation using Gemini.

        Args:
            prompts: A list of text prompts to classify.
            model: The name of the model to use for classification ({VALID_MODELS} are currently supported).

        Returns:
            A list of classification results (the format depends on the model used).

        Raises:
            ValueError: If an unsupported model is specified.
        """
        if model not in VALID_MODELS:
            raise ValueError(f"Unsupported model: {VALID_MODELS} - has to be one ")
        elif model == "gemini":
            generate_func = self.__gemini
        elif model == "claude":
            generate_func = self.__claude

        

        tasks = []
        for q in prompts:
            tasks.append(asyncio.create_task(generate_func(q)))
        
        results = await asyncio.gather(*tasks) # the order of result values is preserved, but the execution order is not. https://docs.python.org/3/library/asyncio-task.html#running-tasks-concurrently
        print(results)
        return results
        

class Evaluate():

    @staticmethod
    def calculate_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        return {"accuracy": accuracy_score(y_true, y_pred),
                       "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
                       "confusion_matrix": confusion_matrix(y_true, y_pred)
                       }
    
    @staticmethod
    def __get_majority_vote(list_of_labels: List[List[int]]) -> List[int]:
        """
        Finds the majority value for each element across multiple lists.

        Args:
            list_of_labels: A list containing the multiple lists  of labels generated by models to compare.

        Returns:
            A list of majority values corresponding to each element position. 
        """
        majority_values = []
        # Iterate over the zipped lists to get elements at the same positions across all lists.
        for elements in zip(*list_of_labels):
            # Create a Counter to count the frequency of each element.
            element_counts = Counter(elements)
            # Get the most common element (with frequency)
            most_common_element = element_counts.most_common(1)[0]
            # Extract the element itself and append it to the results list.
            majority_values.append(most_common_element[0])
        return majority_values
    
    @staticmethod
    def __element_wise_comparison(generated_labels, target_labels):
        result = []
        for sublist in generated_labels:
                comparison_result = [x == y for x, y in zip(sublist, target_labels)]
                result.append(comparison_result) 
        return result

    @staticmethod
    def __graph_label_agreement(arr, y_labels):
        """Visualizes a 2D boolean array using a heatmap.

        Args:
            arr: The 2D boolean array.
            title: Title for the plot (optional).
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        colors = ["lightblue", "lightcoral"]  # Replace with desired colors
        sns.set_palette(sns.color_palette(colors))

        sns.heatmap(arr,
                    cmap=colors, 
                    cbar=True, 
                    cbar_kws={'ticks': [0, 1], 'label': 'Legend', "shrink": 0.2},
                    annot=False, 
                    fmt="d",
                    linewidths=1,
                    yticklabels=y_labels)

        plt.title("Generated Labels vs Aggregated")
        plt.yticks(rotation=45)
        plt.xlabel("Data points")
        plt.ylabel("Generated Lables")
        # Grey out the bottom row
        # plt.gca().add_patch(plt.Rectangle((0, len(comparison_matrix)-1), len(comparison_matrix[0]), 1, fc='grey', alpha=1))
        plt.tight_layout()
        plt.show()


    def classification(self, generated_labels, strategy="majority", visualize=False, y_labels=None):

        if strategy == "majority":
            agg = self.__get_majority_vote(generated_labels)
        
        m = self.__element_wise_comparison(generated_labels, agg)

        if visualize and y_labels is not None:
            self.__graph_label_agreement(m, y_labels)
            return m
        elif visualize and y_labels is None:
            raise ValueError(f"Pass y_labels when visualize==True")
        else:
            return m



            





if __name__ == "__main__":
    ann = Annotate()
    import time
    s = time.perf_counter()
    prompts = ["where is paris?", "whare is tehran"]
    asyncio.run(ann.TextClassification(prompts, model="gemini"))
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")


