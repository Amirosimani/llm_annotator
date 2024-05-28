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


from config import VALID_MODELS, GEMINI_CONFIG, CLAUDE_CONFIG, PALM_CONFIG, VERBOSE


def init_logger():
    global logger
    logger = logging.getLogger('base')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

init_logger()


class Annotate:
    def __init__(self, gemini_config=GEMINI_CONFIG, claude_config=CLAUDE_CONFIG, palm_config=PALM_CONFIG, verbose=VERBOSE):

        self.gemini_config = gemini_config
        self.claude_config = claude_config
        self.palm_config = palm_config


         # Initialize logger with the desired level based on verbose setting
        self.logger = logging.getLogger('Annotate')
        self.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)  

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

        from vertexai.generative_models import GenerativeModel
        import vertexai.preview.generative_models as generative_models

        vertexai.init(project=self.gemini_config["project_config"]["project"], 
                      location=self.gemini_config["project_config"]["location"])

        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        rate_limiter = Limiter(self.gemini_config["project_config"]["qpm"]/60) # Limit to 300 requests per 60 second
        model = GenerativeModel(self.gemini_config["model"],
                                system_instruction=[
                                "You are a helpful data labeler.",
                                "Your mission is to label the input data based on the instruction you will receive.",
                                ],
                            )
        await rate_limiter.wait()
        try:
            self.logger.debug(f"Sending prompt to Gemini: {prompt}")    
            responses = model.generate_content(
                [prompt],
                generation_config=self.gemini_config["generation_config"],
                safety_settings=safety_settings,
                stream=False,
                )
        except Exception as e:
            self.logger.error(f"Error in __gemini: {e}") 
            raise
        self.logger.info(f"Gemini response received.")  # Info log
        return(self.__extract_binary_values(responses.text))

    
    async def __palm(self, prompt:str) -> List:
        """
        Asynchronously generates labels for datapoints using Bison model 
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

        from vertexai.language_models import TextGenerationModel

        vertexai.init(project=self.palm_config["project_config"]["project"], 
                      location=self.palm_config["project_config"]["location"])


        rate_limiter = Limiter(self.palm_config["project_config"]["qpm"]/60) # Limit to 300 requests per 60 second
        model = TextGenerationModel.from_pretrained(self.palm_config["model"])
        await rate_limiter.wait()
        try:
            self.logger.debug(f"Sending prompt to PaLM: {prompt}")    
            responses = model.predcit(
                prompt,
                **self.palm_config["generation_config"]
                )
        except Exception as e:
            self.logger.error(f"Error in __palm: {e}") 
            raise
        self.logger.info(f"palm response received.")  # Info log
        return(self.__extract_binary_values(responses.text))
    

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
        from anthropic import AnthropicVertex

        client = AnthropicVertex(region=self.claude_config["project_config"]["location"], 
                                 project_id=self.claude_config["project_config"]["project"])


        rate_limiter = Limiter(self.claude_config["project_config"]["qpm"]/60) # Limit to 60 requests per 60 second

        await rate_limiter.wait()
        try:
            self.logger.debug(f"Sending prompt to Claude: {prompt}")
            responses = client.messages.create(
                max_tokens=1024,
                messages=[
                    {"role": "user",
                    "content": prompt,
                    }
                    ],
                    model=self.claude_config["model"],
                    )
        except Exception as e:
            self.logger.error(f"Error in __claude: {e}")
            raise
        self.logger.info(f"Claude response received.")  # Info log
        return(self.__extract_binary_values(responses.content[0].text))

    async def classification(self, prompts: List[str], models: List[str], valid_models=VALID_MODELS) ->  List[Any]:
        f"""
        Performs text classification labeling

        Args:
            prompts: A list of text prompts to classify.
            models: The name of the model to use for classification ({valid_models} are currently supported).
            valid_models: A list of valid model names.


        Returns:
            A list of classification results (the format depends on the model used).

        Raises:
            ValueError: If an unsupported model is specified.
        """

        if not all(model in valid_models for model in models):
            self.logger.error(f"Unsupported model: {set(models) - set(valid_models)}")
            raise ValueError(f"Unsupported models in valid_models. Please use models from {valid_models}")


        model_to_method = {
            "claude": self.__claude,
            "gemini": self.__gemini,
            "palm": self.__palm
            # Add more mappings as needed for other models
        }

        generate_methods = [model_to_method[model] for model in models]
        all_tasks = {model: [] for model in models}

        # Progress bar for creating tasks
        total_tasks = len(prompts) * len(models)
        with tqdm_asyncio(total=total_tasks, desc="Creating tasks") as pbar:  # Use tqdm_asyncio for progress bar
            for i, q in enumerate(prompts):
            # Dynamically select the correct method based on the model
                for m in models:
                    generate_method = generate_methods[models.index(m)]
                    all_tasks[m].append(asyncio.create_task(generate_method(q)))
                    pbar.update(2)  # Update progress bar for both tasks

        for method_name, tasks in all_tasks.items():
            with tqdm_asyncio(total=len(tasks), desc=f"Gathering {method_name} results") as pbar:  # Progress bar for gathering results
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"{method_name} Task {i} failed: {result}")
                        all_tasks[method_name][i] = None  # Store None for failed tasks
                    else:
                        all_tasks[method_name][i] = result
                    pbar.update(1)  # Update the progress bar
                self.logger.debug(f"Classification results for {method_name}: {all_tasks[method_name]}")

        return all_tasks



        
class Aggregate():

    @staticmethod
    def _get_majority_vote(list_of_labels: List[List[int]]) -> List[int]:
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
    
    def _glad(self, list_of_labels: List[List[int]], verbose=VERBOSE) -> List[int]:

        THRESHOLD = 1e-5

        def glad_dataset(**kwargs):
            """Function to create a dataset-like dictionary."""
            return kwargs

        def load_data(arr):
            data = glad_dataset(
                numLabels=sum(len(sublist) for sublist in arr),
                numLabelers=len(arr),
                numTasks=len(arr[0]),
                numClasses=len(set(arr[0])),
                priorZ=np.repeat(1 / len(set(arr[0])), len(set(arr[0]))),
                labels=np.array(list(map(list, zip(*arr))))
            )

            assert np.isclose(data['priorZ'].sum(), 1), 'Incorrect priorZ given'

            data['priorAlpha'] = np.ones(data['numLabelers'])
            data['priorBeta'] = np.ones(data['numTasks'])
            data['probZ'] = np.empty((data['numTasks'], data['numClasses']))
            data['beta'] = np.empty(data['numTasks'])
            data['alpha'] = np.empty(data['numLabelers'])

            return data
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def logsigmoid(x):
            return - np.logaddexp(0, -x)
        

        def EM(data):
            data["alpha"] = data["priorAlpha"].copy()
            data["beta"] = data["priorBeta"].copy()
            data["probZ"][:] = data["priorZ"][:]

            print(data["probZ"][:])

            EStep(data)
            lastQ = computeQ(data)
            MStep(data)
            Q = computeQ(data)
            counter = 1
            while abs((Q - lastQ) / lastQ) > THRESHOLD:
                if verbose:
                    logger.info('EM: iter={}'.format(counter))
                lastQ = Q
                EStep(data)
                MStep(data)
                Q = computeQ(data)
                counter += 1

        def calcLogProbL(item, *args):
            data = args[-1]
            print(data["alpha"], data["labels"])

            j = int(item[0])
            delta = args[0][j]
            noResp = args[1][j]
            oneMinusDelta = (~delta) & (~noResp)

            exponents = item[1:]

            correct = logsigmoid(exponents[delta]).sum()
            wrong = (logsigmoid(-exponents[oneMinusDelta]) - np.log(float(data["numClasses"] - 1))).sum()

            return correct + wrong

        def EStep(data):
            data["probZ"] = np.tile(np.log(data["priorZ"]), data["numTasks"]).reshape(data["numTasks"], data["numClasses"])

            ab = np.dot(np.array([np.exp(data["beta"])]).T, np.array([data["alpha"]]))
            ab = np.c_[np.arange(data["numTasks"]), ab]
            for k in range(data["numClasses"]):
                data["probZ"][:, k] = np.apply_along_axis(calcLogProbL, 1, ab,
                                                    (data["labels"] == k + 1),
                                                    (data["labels"] == 0),
                                                    data)  # Pass data as an additional argument

            data["probZ"] = np.exp(data["probZ"])
            s = data["probZ"].sum(axis=1)
            data["probZ"] = (data["probZ"].T / s).T
            assert not np.any(np.isnan(data["probZ"])), 'Invalid Value [EStep]'



        def df(x, *args):
            data = args[0]
            d = glad_dataset(labels=data["labels"], numLabels=data["numLabels"], numLabelers=data["numLabelers"],
                        numTasks=data["numTasks"], numClasses=data["numClasses"],
                        priorAlpha=data["priorAlpha"], priorBeta=data["priorBeta"],
                        priorZ=data["priorZ"], probZ=data["probZ"])
            unpackX(x, d)
            dQdAlpha, dQdBeta = gradientQ(d)
            return np.r_[-dQdAlpha, -dQdBeta]


        def f(x, *args):
            u"""Return the value of the objective function
            """
            data = args[0]
            d = glad_dataset(labels=data["labels"], numLabels=data["numLabels"], numLabelers=data["numLabelers"],
                        numTasks=data["numTasks"], numClasses=data["numClasses"],
                        priorAlpha=data["priorAlpha"], priorBeta=data["priorBeta"],
                        priorZ=data["priorZ"], probZ=data["probZ"])
            unpackX(x, d)
            return - computeQ(d)

        def MStep(data):
            initial_params = packX(data)
            params = sp.optimize.minimize(fun=f, x0=initial_params, args=(data,), method='CG',
                                        jac=df, tol=0.01,
                                        options={'maxiter': 25, 'disp': VERBOSE})
            unpackX(params.x, data)


        def computeQ(data):
            Q = 0
            Q += (data["probZ"] * np.log(data["priorZ"])).sum()

            ab = np.dot(np.array([np.exp(data["beta"])]).T, np.array([data["alpha"]]))

            logSigma = logsigmoid(ab)
            idxna = np.isnan(logSigma)
            if np.any(idxna):
                logger.warning('an invalid value was assigned to np.log [computeQ]')
                logSigma[idxna] = ab[idxna]

            logOneMinusSigma = logsigmoid(-ab) - np.log(float(data["numClasses"] - 1))
            idxna = np.isnan(logOneMinusSigma)
            if np.any(idxna):
                logger.warning('an invalid value was assigned to np.log [computeQ]')
                logOneMinusSigma[idxna] = -ab[idxna]

            for k in range(data["numClasses"]):
                delta = (data["labels"] == k + 1)
                Q += (data["probZ"][:, k] * logSigma.T).T[delta].sum()
                oneMinusDelta = (data["labels"] != k + 1) & (data["labels"] != 0)
                Q += (data["probZ"][:, k] * logOneMinusSigma.T).T[oneMinusDelta].sum()

            Q += np.log(sp.stats.norm.pdf(data["alpha"] - data["priorAlpha"])).sum()
            Q += np.log(sp.stats.norm.pdf(data["beta"] - data["priorBeta"])).sum()

            if np.isnan(Q):
                return -np.inf
            return Q


        def gradientQ(data):
            dQdAlpha = - (data["alpha"] - data["priorAlpha"])
            dQdBeta = - (data["beta"] - data["priorBeta"])

            ab = np.exp(data["beta"])[:, np.newaxis] * data["alpha"]
            sigma = sigmoid(ab)
            sigma[np.isnan(sigma)] = 0

            for k in range(data["numClasses"]):
                delta = (data["labels"] == k + 1)
                oneMinusDelta = (data["labels"] != k + 1) & (data["labels"] != 0)

                dQdAlpha += (data["probZ"][:, k][:, np.newaxis] * np.exp(data["beta"])[:, np.newaxis] * (delta - sigma)).sum(axis=0)
                dQdBeta += (data["probZ"][:, k][:, np.newaxis] * data["alpha"] * (delta - sigma)).sum(axis=1)

            return dQdAlpha, dQdBeta


        def dAlpha(item, *args):
            i = int(item[0])
            sigma_ab = item[1:]

            delta = args[0][:, i]
            noResp = args[1][:, i]
            oneMinusDelta = (~delta) & (~noResp)

            probZ = args[2]

            data = args[3] 

            correct = probZ[delta] * np.exp(data["beta"][delta]) * (1 - sigma_ab[delta])
            wrong = probZ[oneMinusDelta] * np.exp(data["beta"][oneMinusDelta]) * (-sigma_ab[oneMinusDelta])

            return correct.sum() + wrong.sum()


        def dBeta(item, *args):
            j = int(item[0])
            sigma_ab = item[1:]

            delta = args[0][j]
            noResp = args[1][j]
            oneMinusDelta = (~delta) & (~noResp)

            probZ = args[2][j]
            data = args[3] 

            correct = probZ * data["alpha"][delta] * (1 - sigma_ab[delta])
            wrong = probZ * data["alpha"][oneMinusDelta] * (-sigma_ab[oneMinusDelta])

            return correct.sum() + wrong.sum()


        def packX(data):
            return np.r_[data["alpha"].copy(), data["beta"].copy()]


        def unpackX(x, data):
            data["alpha"] = x[:data["numLabelers"]].copy()
            data["beta"] = x[data["numLabelers"]:].copy()


        def output(data):
            alpha = np.c_[np.arange(data["numLabelers"]), data["alpha"]]
            beta = np.c_[np.arange(data["numTasks"]), np.exp(data["beta"])]
            probZ = np.c_[np.arange(data["numTasks"]), data["probZ"]]
            label = np.c_[np.arange(data["numTasks"]), np.argmax(data["probZ"], axis=1)]

            return {"alpha": alpha,
                    "beta": beta,
                    "probZ": probZ,
                    "labels": label}
        

        data = load_data(list_of_labels)
        EM(data)
        return output(data)


class Evaluate():

    def __init__(self,verbose=VERBOSE):

         # Initialize logger with the desired level based on verbose setting
        self.logger = logging.getLogger('Annotate')
        self.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)  

    @staticmethod
    def calculate_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        return {"accuracy": accuracy_score(y_true, y_pred),
                       "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
                       "confusion_matrix": confusion_matrix(y_true, y_pred)
                       }
    
    
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
        self.logger.debug(f"Evaluating classification with strategy: {strategy}")

        if strategy == "majority":
            agg = Aggregate()._get_majority_vote(generated_labels)
        elif strategy == "glad":
            agg = Aggregate()._glad(generated_labels)
        
        m = self.__element_wise_comparison(generated_labels, agg)
        self.logger.info(f"Evaluation complete.")
        if visualize and y_labels is not None:
            self.__graph_label_agreement(m, y_labels)
            return m
        elif visualize and y_labels is None:
            raise ValueError(f"Pass y_labels when visualize==True")
        else:
            return m



            





if __name__ == "__main__":
    ann = Annotate(verbose=VERBOSE)
    import time
    s = time.perf_counter()
    prompts = ["where is paris?", "whare is tehran"]
    asyncio.run(ann.classification(prompts, model="gemini"))
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")


