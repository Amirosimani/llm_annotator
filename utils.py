import os
import logging
import asyncio
import numpy as np
import scipy as sp
from collections import Counter
from datetime import datetime
from asynciolimiter import Limiter
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List


import vertexai


from config import VALID_MODELS, GEMINI_CONFIG, PALM_CONFIG, VERBOSE


def init_logger(verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

init_logger()

THRESHOLD = 1e-5
VERBOSE = False
DEBUG = False
LOGGER = None


class Annotate:
    def __init__(self, verbose=VERBOSE):

         # Initialize logger with the desired level based on verbose setting
        self.logger = logging.getLogger('Annotate')
        self.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)  


    async def __gemini(self, prompt:str, gemini_config: dict) -> List:
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

        vertexai.init(project=gemini_config["project_config"]["project"], 
                      location=gemini_config["project_config"]["location"])

        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

        rate_limiter = Limiter(gemini_config["project_config"]["qpm"]/60) # Limit to 300 requests per 60 second
        model = GenerativeModel(gemini_config["model"],
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
                generation_config=gemini_config["generation_config"],
                safety_settings=safety_settings,
                stream=False,
                )
        except Exception as e:
            self.logger.error(f"Error in __gemini: {e}") 
            raise
        self.logger.info(f"Gemini response received.")  # Info log
        return(responses.text)
    
    async def __palm(self, prompt:str, palm_config: dict) -> List:
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

        vertexai.init(project=palm_config["project_config"]["project"], 
                      location=palm_config["project_config"]["location"])


        rate_limiter = Limiter(palm_config["project_config"]["qpm"]/60) # Limit to 300 requests per 60 second
        model = TextGenerationModel.from_pretrained(palm_config["model"])
        await rate_limiter.wait()
        try:
            self.logger.debug(f"Sending prompt to PaLM: {prompt}")    
            responses = model.predict(
                prompt,
                **palm_config["generation_config"]
                )
        except Exception as e:
            self.logger.error(f"Error in __palm: {e}") 
            raise
        self.logger.info(f"palm response received.")  # Info log
        return(responses.text)
    
    async def __claude(self, prompt:str, claude_config: dict) -> List:
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

        client = AnthropicVertex(region=claude_config["project_config"]["location"], 
                                 project_id=claude_config["project_config"]["project"])


        rate_limiter = Limiter(claude_config["project_config"]["qpm"]/60) # Limit to 60 requests per 60 second

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
                    model=claude_config["model"],
                    )
        except Exception as e:
            self.logger.error(f"Error in __claude: {e}")
            raise
        self.logger.info(f"Claude response received.")  # Info log
        return(responses.content[0].text)

    async def classification(self, prompts: List[str], models: List[str], model_configs,
                             valid_models=VALID_MODELS) -> List[Any]:
        """Performs text classification labeling.

        Args:
            prompts: A list of text prompts to classify.
            models: The names of the models to use for classification.
            valid_models: A list of valid model names.
            gemini_config: (Optional) Gemini configuration if "gemini" is in `models`.
            palm_config: (Optional) PaLM configuration if "palm" is in `models`.

        Returns:
            A list of classification results for each model (None for failed tasks).

        Raises:
            ValueError: If an unsupported model or missing configuration is specified.
        """
        if not all(model in valid_models for model in models):
            self.logger.error(f"Unsupported model: {set(models) - set(valid_models)}")
            raise ValueError(f"Unsupported models in valid_models. Please use models from {valid_models}")

        # Create tasks for each model and config combination
        all_tasks = {}
        total_tasks = len(prompts) * sum(len(configs) for model, configs in model_configs.items() if model in models)

        with tqdm_asyncio(total=total_tasks, desc="Creating tasks") as pbar:
            for prompt in prompts:
                for model in models:
                    configs = model_configs[model] if isinstance(model_configs[model], list) else [model_configs[model]]
                    for config in configs:
                        task_key = f"{model}_{config['config_name']}"  # Assuming each config has a unique 'config_name' key
                        if task_key not in all_tasks:
                            all_tasks[task_key] = []
                        if model == "gemini":
                            all_tasks[task_key].append(asyncio.create_task(self.__gemini(prompt, config)))
                        elif model == "palm":
                            all_tasks[task_key].append(asyncio.create_task(self.__palm(prompt, config)))
                        # Add other models here as needed
                        pbar.update(1)  # Update progress bar for each created task

        # Collect results for each model
        results = {}
        for task_key, tasks in all_tasks.items():
            with tqdm_asyncio(total=len(tasks), desc=f"Gathering {task_key} results") as pbar:
                model_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(model_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"{task_key} Task {i} failed: {result}")
                        model_results[i] = None  # Store None for failed tasks
                    pbar.update(1)  # Update the progress bar

                self.logger.debug(f"Classification results for {task_key}: {model_results}")
                results[task_key] = model_results

        return results


class Aggregate:
    def __init__(self, verbose=VERBOSE):

         # Initialize logger with the desired level based on verbose setting
        self.logger = logging.getLogger('Aggregate')
        self.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)  

    @staticmethod
    def majority_vote(label_dict: Dict[str, List[int]]) -> List[int]:
        """
        Finds the majority value for each element across multiple lists within a dictionary.

        Args:
            label_dict: A dictionary where keys are identifiers and values are lists of labels.

        Returns:
            A list of majority values corresponding to each element position.
    """
        list_of_labels = list(label_dict.values())  # Extract values into a list
        majority_values = []

        for elements in zip(*list_of_labels):
            element_counts = Counter(elements)
            most_common_element = element_counts.most_common(1)[0]
            majority_values.append(most_common_element[0])

        return majority_values

    class Dataset:
        def __init__(self, labels=None,
                     numLabels=-1, numLabelers=-1, numTasks=-1, numClasses=-1,
                     priorAlpha=None, priorBeta=None, priorZ=None,
                     alpha=None, beta=None, probZ=None):
            self.labels = labels
            self.numLabels = numLabels
            self.numLabelers = numLabelers
            self.numTasks = numTasks
            self.numClasses = numClasses
            self.priorAlpha = priorAlpha
            self.priorBeta = priorBeta
            self.priorZ = priorZ
            self.alpha = alpha
            self.beta = beta
            self.probZ = probZ

    @staticmethod
    def convert_dict_to_indexed_list(data_dict):
        number_map = {key: index for index, key in enumerate(data_dict.keys())}
        max_len = len(next(iter(data_dict.values())))

        result = []
        for index in range(max_len):
            for key, value_list in data_dict.items():
                value = value_list[index]
                # Convert to 0 if not an integer
                converted_value = 0 if not isinstance(value, int) else value
                result.append([index, number_map[key], converted_value])
        return result

    @staticmethod
    def load_data(filename):
        data = Aggregate.Dataset()
        with open(filename) as f:
            # Read parameters
            header = f.readline().split()
            data.numLabels = int(header[0])
            data.numLabelers = int(header[1])
            data.numTasks = int(header[2])
            data.numClasses = int(header[3])
            data.priorZ = np.array([float(x) for x in header[4:]])
            assert len(data.priorZ) == data.numClasses, 'Incorrect input header'
            assert data.priorZ.sum() == 1, 'Incorrect priorZ given'
            logging.debug('Reading {} labels of {} labelers over {} tasks for prior P(Z) = {}'.format(data.numLabels,
                                                                                                    data.numLabelers,
                                                                                                    data.numTasks,
                                                                                                    data.priorZ))
            # Read Labels
            data.labels = np.zeros((data.numTasks, data.numLabelers))
            for line in f:
                task, labeler, label = map(int, line.split())
                logging.debug("Read: task({})={} by labeler {}".format(task, label, labeler))
                data.labels[task][labeler] = label + 1
        # Initialize Probs
        data.priorAlpha = np.ones(data.numLabelers)
        data.priorBeta = np.ones(data.numTasks)
        data.probZ = np.empty((data.numTasks, data.numClasses))
        data.beta = np.empty(data.numTasks)
        data.alpha = np.empty(data.numLabelers)

        return data

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def logsigmoid(x):
        return - np.log(1 + np.exp(-x))

    @staticmethod
    def EM(data):
        u"""Infer true labels, tasks' difficulty and workers' ability
        """
        # Initialize parameters to starting values
        data.alpha = data.priorAlpha.copy()
        data.beta = data.priorBeta.copy()
        data.probZ[:] = data.priorZ[:]

        Aggregate.EStep(data)
        lastQ = Aggregate.computeQ(data)
        Aggregate.MStep(data)
        Q = Aggregate.computeQ(data)
        counter = 1
        while abs((Q - lastQ) / lastQ) > THRESHOLD:
            logging.info('EM: iter={}'.format(counter))
            lastQ = Q
            Aggregate.EStep(data)
            Aggregate.MStep(data)
            Q = Aggregate.computeQ(data)
            counter += 1

    @staticmethod
    def EStep(data):
        u"""Evaluate the posterior probability of true labels given observed labels and parameters
        """

        def calcLogProbL(item, *args):
            j = int(item[0])  # task ID

            # List[boolean]: denotes if the worker i picked the focused class for the task j
            ## formally, delta[i, j] = True if l_ij == z_j for i = 0, ..., m-1 (m=# of workers)
            delta = args[0][j]
            noResp = args[1][j]
            oneMinusDelta = (~delta) & (~noResp)

            # List[float]: alpha_i * exp(beta_j) for i = 0, ..., m-1
            exponents = item[1:]

            # Log likelihood for the observations s.t. l_ij == z_j
            correct = Aggregate.logsigmoid(exponents[delta]).sum()
            # Log likelihood for the observations s.t. l_ij != z_j
            wrong = (Aggregate.logsigmoid(-exponents[oneMinusDelta]) - np.log(float(data.numClasses - 1))).sum()

            # Return log likelihood
            return correct + wrong

        logging.info('EStep')
        data.probZ = np.tile(np.log(data.priorZ), data.numTasks).reshape(data.numTasks, data.numClasses)

        ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))
        ab = np.c_[np.arange(data.numTasks), ab]

        for k in range(data.numClasses):
            data.probZ[:, k] = np.apply_along_axis(calcLogProbL, 1, ab,
                                                   (data.labels == k + 1),
                                                   (data.labels == 0))

        # Exponentiate and renormalize
        data.probZ = np.exp(data.probZ)
        s = data.probZ.sum(axis=1)
        data.probZ = (data.probZ.T / s).T
        assert not np.any(np.isnan(data.probZ)), 'Invalid Value [EStep]'
        assert not np.any(np.isinf(data.probZ)), 'Invalid Value [EStep]'

        return data

    @staticmethod
    def packX(data):
        return np.r_[data.alpha.copy(), data.beta.copy()]

    @staticmethod
    def unpackX(x, data):
        data.alpha = x[:data.numLabelers].copy()
        data.beta = x[data.numLabelers:].copy()

    @staticmethod
    def getBoundsX(data, alpha=(-100, 100), beta=(-100, 100)):
        alpha_bounds = np.array([[alpha[0], alpha[1]] for i in range(data.numLabelers)])
        beta_bounds = np.array([[beta[0], beta[1]] for i in range(data.numLabelers)])
        return np.r_[alpha_bounds, beta_bounds]

    @staticmethod
    def f(x, *args):
        u"""Return the value of the objective function
        """
        data = args[0]
        d = Aggregate.Dataset(labels=data.labels, numLabels=data.numLabels, numLabelers=data.numLabelers,
                    numTasks=data.numTasks, numClasses=data.numClasses,
                    priorAlpha=data.priorAlpha, priorBeta=data.priorBeta,
                    priorZ=data.priorZ, probZ=data.probZ)
        Aggregate.unpackX(x, d)
        return - Aggregate.computeQ(d)

    @staticmethod
    def df(x, *args):
        u"""Return gradient vector
        """
        data = args[0]
        d = Aggregate.Dataset(labels=data.labels, numLabels=data.numLabels, numLabelers=data.numLabelers,
                    numTasks=data.numTasks, numClasses=data.numClasses,
                    priorAlpha=data.priorAlpha, priorBeta=data.priorBeta,
                    priorZ=data.priorZ, probZ=data.probZ)
        Aggregate.unpackX(x, d)
        dQdAlpha, dQdBeta = Aggregate.gradientQ(d)
        # Flip the sign since we want to minimize
        assert not np.any(np.isinf(dQdAlpha)), 'Invalid Gradient Value [Alpha]'
        assert not np.any(np.isinf(dQdBeta)), 'Invalid Gradient Value [Beta]'
        assert not np.any(np.isnan(dQdAlpha)), 'Invalid Gradient Value [Alpha]'
        assert not np.any(np.isnan(dQdBeta)), 'Invalid Gradient Value [Beta]'
        return np.r_[-dQdAlpha, -dQdBeta]

    @staticmethod
    def MStep(data, verbose=False):
        logging.info('MStep')
        initial_params = Aggregate.packX(data)
        params = sp.optimize.minimize(fun=Aggregate.f, x0=initial_params, args=(data,), method='CG',
                                      jac=Aggregate.df, tol=0.01,
                                      options={'maxiter': 25, 'disp': verbose})
        logging.debug(params)
        Aggregate.unpackX(params.x, data)

    @staticmethod
    def computeQ(data):
        u"""Calculate the expectation of the joint likelihood
        """
        Q = 0
        # Start with the expectation of the sum of priors over all tasks
        Q += (data.probZ * np.log(data.priorZ)).sum()

        # the expectation of the sum of posteriors over all tasks
        ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))

        # logSigma = - np.log(1 + np.exp(-ab))
        logSigma = Aggregate.logsigmoid(ab)  # logP
        idxna = np.isnan(logSigma)
        if np.any(idxna):
            logging.warning('an invalid value was assigned to np.log [computeQ]')
            logSigma[idxna] = ab[idxna]  # For large negative x, -log(1 + exp(-x)) = x

        # logOneMinusSigma = - np.log(1 + np.exp(ab))
        logOneMinusSigma = Aggregate.logsigmoid(-ab) - np.log(float(data.numClasses - 1))  # log((1-P)/(K-1))
        idxna = np.isnan(logOneMinusSigma)
        if np.any(idxna):
            logging.warning('an invalid value was assigned to np.log [computeQ]')
            logOneMinusSigma[idxna] = -ab[idxna]  # For large positive x, -log(1 + exp(x)) = x

        for k in range(data.numClasses):
            delta = (data.labels == k + 1)
            Q += (data.probZ[:, k] * logSigma.T).T[delta].sum()
            oneMinusDelta = (data.labels != k + 1) & (data.labels != 0)  # label == 0 -> no response
            Q += (data.probZ[:, k] * logOneMinusSigma.T).T[oneMinusDelta].sum()

        # Add Gaussian (standard normal) prior for alpha
        Q += np.log(sp.stats.norm.pdf(data.alpha - data.priorAlpha)).sum()

        # Add Gaussian (standard normal) prior for beta
        Q += np.log(sp.stats.norm.pdf(data.beta - data.priorBeta)).sum()

        logging.debug('a[0]={} a[1]={} a[2]={} b[0]={}'.format(data.alpha[0], data.alpha[1],
                                                            data.alpha[2], data.beta[0]))
        logging.debug('Q={}'.format(Q))
        if np.isnan(Q):
            return -np.inf
        return Q
    

    @staticmethod
    def gradientQ(data):
        def dAlpha(item, *args):
            i = int(item[0])  # worker ID
            sigma_ab = item[1:] # List[float], dim=(n,): sigmoid(alpha_i * beta_j) for j = 0, ..., n-1

            # List[boolean], dim=(n,): denotes if the worker i picked the focused class for
            # task j (j=0, ..., n-1)
            delta = args[0][:, i]
            noResp = args[1][:, i]
            oneMinusDelta = (~delta) & (~noResp)

            # List[float], dim=(n,): Prob of the true label of the task j being the focused class (p^k)
            probZ = args[2]

            correct = probZ[delta] * np.exp(data.beta[delta]) * (1 - sigma_ab[delta])
            wrong = probZ[oneMinusDelta] * np.exp(data.beta[oneMinusDelta]) * (-sigma_ab[oneMinusDelta])
            # Note: The derivative in Whitehill et al.'s appendix has the term ln(K-1), which is incorrect.

            return correct.sum() + wrong.sum()

        def dBeta(item, *args):
            j = int(item[0])  # task ID
            sigma_ab = item[1:] # List[float], dim=(m,): sigmoid(alpha_i * beta_j) for i = 0, ..., m-1

            # List[boolean], dim=(m,): denotes if the worker i picked the focused class for
            # task j (i=0, ..., m-1)
            delta = args[0][j]
            noResp = args[1][j]
            oneMinusDelta = (~delta) & (~noResp)

            # float: Prob of the true label of the task j being the focused class (p^k)
            probZ = args[2][j]

            correct = probZ * data.alpha[delta] * (1 - sigma_ab[delta])
            wrong = probZ * data.alpha[oneMinusDelta] * (-sigma_ab[oneMinusDelta])

            return correct.sum() + wrong.sum()

        # prior prob.
        dQdAlpha = - (data.alpha - data.priorAlpha)
        dQdBeta = - (data.beta - data.priorBeta)

        ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))

        sigma = Aggregate.sigmoid(ab)
        sigma[np.isnan(sigma)] = 0  # :TODO check if this is correct

        labelersIdx = np.arange(data.numLabelers).reshape((1, data.numLabelers))
        sigma = np.r_[labelersIdx, sigma]
        sigma = np.c_[np.arange(-1, data.numTasks), sigma]
        # sigma: List[List[float]]: dim=(n+1, m+1) where n = # of tasks and m = # of workers
        # sigma[0] = List[float]: worker IDs (-1, 0, ..., m-1) where the first -1 is a pad
        # sigma[:, 0] = List[float]: task IDs (-1, 0, ..., n-1) where the first -1 is a pad

        for k in range(data.numClasses):
            dQdAlpha += np.apply_along_axis(dAlpha, 0, sigma[:, 1:],
                                            (data.labels == k + 1),
                                            (data.labels == 0),
                                            data.probZ[:, k])

            dQdBeta += np.apply_along_axis(dBeta, 1, sigma[1:],
                                        (data.labels == k + 1),
                                            (data.labels == 0),
                                        data.probZ[:, k]) * np.exp(data.beta)

        logging.debug('dQdAlpha[0]={} dQdAlpha[1]={} dQdAlpha[2]={} dQdBeta[0]={}'.format(dQdAlpha[0], dQdAlpha[1],
                                                                                            dQdAlpha[2], dQdBeta[0]))
        return dQdAlpha, dQdBeta
    
    @staticmethod
    def output(data, save):
        results = {}
        # Alpha (worker abilities)
        results['alpha'] = {i: data.alpha[i] for i in range(data.numLabelers)}
        # Beta (task difficulties) - Convert to probabilities for consistency
        results['beta'] = {j: np.exp(data.beta[j]) for j in range(data.numTasks)}
        # ProbZ (posterior probabilities for each task and class)
        results['probZ'] = {}
        for j in range(data.numTasks):
            results['probZ'][j] = {k: data.probZ[j, k] for k in range(data.numClasses)}
        # Predicted Labels
        results['labels'] = {j: np.argmax(data.probZ[j]) for j in range(data.numTasks)}

        if save:
            now = datetime.now().strftime("%Y%m%d")
            try:
                os.mkdir(f"../data/{now}/")
            except:
                pass
            alpha = np.c_[np.arange(data.numLabelers), data.alpha]
            np.savetxt(f'../data/{now}/alpha__{now}.csv', alpha, fmt=['%d', '%.5f'], delimiter=',', header='id,alpha')
            beta = np.c_[np.arange(data.numTasks), np.exp(data.beta)]
            np.savetxt(f'../data/{now}/beta__{now}.csv', beta, fmt=['%d', '%.5f'], delimiter=',', header='id,beta')
            probZ = np.c_[np.arange(data.numTasks), data.probZ]
            np.savetxt(fname=f'../data/{now}/probZ__{now}.csv',
                    X=probZ,
                    fmt=['%d'] + (['%.5f'] * data.numClasses),
                    delimiter=',',
                    header='id,' + ','.join(['z' + str(k) for k in range(data.numClasses)]))
            label = np.c_[np.arange(data.numTasks), np.argmax(data.probZ, axis=1)]
            np.savetxt(f'../data/{now}label_glad__{now}.csv', label, fmt=['%d', '%d'], delimiter=',', header='id,label')

        return results
    

    @staticmethod
    def glad(filename, save=False):
        # data = load_data(data, task_config)
        data = Aggregate.load_data(filename)
        Aggregate.EM(data)

        return Aggregate.output(data, save)


class Evaluate:
    def __init__(self, verbose=False):
        # Initialize logger with the desired level based on verbose setting
        self.logger = logging.getLogger('Aggregate')
        self.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)

    def evaluate_entity_extraction(self, predicted_entities, true_entities):
        """
        Evaluate entity extraction performance using precision, recall, F1 score, and exact match ratio.

        Parameters:
        - predicted_entities: A list of predicted entities, where each entity is represented by a dictionary.
        - true_entities: A list of true entities, where each entity is represented by a dictionary.

        Returns:
        - A dictionary containing precision, recall, F1 score, and exact match ratio.
        """
        # Flatten the predicted and true entities
        predicted_flat = [entity for sublist in predicted_entities for entity in sublist]
        true_flat = [entity for sublist in true_entities for entity in sublist]

        # Convert to sets for easier comparison (case insensitive)
        predicted_set = {(ent['text'][0].lower(), ent['type'].lower()) for ent in predicted_flat}
        true_set = {(ent['text'][0].lower(), ent['type'].lower()) for ent in true_flat}

        # Calculate True Positives, False Positives, and False Negatives
        true_positives = len(predicted_set & true_set)
        false_positives = len(predicted_set - true_set)
        false_negatives = len(true_set - predicted_set)

        # Precision, Recall, and F1 Score calculations
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

        # Exact Match Ratio
        exact_match_ratio = len(predicted_set & true_set) / len(true_set) if len(true_set) > 0 else 0.0

        # Results dictionary
        results = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Exact Match Ratio': exact_match_ratio,
            'Total Predicted Entities': len(predicted_flat),
            'Total True Entities': len(true_flat)
        }

        return results

    def bootstrap_evaluation(self, predicted_entities, true_entities, B=500):
        """
        Calculate bootstrap confidence intervals for entity extraction performance metrics.

        Parameters:
        - predicted_entities: A list of predicted entities (nested list format).
        - true_entities: A list of true entities (nested list format).
        - B: Number of bootstrap samples to draw.

        Returns:
        - A dictionary containing confidence intervals for precision, recall, F1 score, and exact match ratio.
        """
        n = len(predicted_entities)
        metrics = np.zeros((B, 4))  # Precision, recall, F1, exact match ratio

        for i in range(B):
            # Resample indices with replacement
            indices = np.random.choice(n, n, replace=True)
            resampled_predicted = [predicted_entities[idx] for idx in indices]
            resampled_true = [true_entities[idx] for idx in indices]

            # Evaluate on resampled data
            result = self.evaluate_entity_extraction(resampled_predicted, resampled_true)
            metrics[i] = [result['Precision'], result['Recall'], result['F1 Score'], result['Exact Match Ratio']]

        # Confidence intervals
        ci_precision = np.percentile(metrics[:, 0], [2.5, 97.5])
        ci_recall = np.percentile(metrics[:, 1], [2.5, 97.5])
        ci_f1 = np.percentile(metrics[:, 2], [2.5, 97.5])
        ci_exact_match = np.percentile(metrics[:, 3], [2.5, 97.5])

        # Results
        confidence_intervals = {
            'Precision CI': ci_precision,
            'Recall CI': ci_recall,
            'F1 Score CI': ci_f1,
            'Exact Match Ratio CI': ci_exact_match
        }

        return confidence_intervals



if __name__ == "__main__":
    ann = Annotate(verbose=VERBOSE)
    import time
    s = time.perf_counter()
    prompts = ["where is paris?", "whare is tehran"]
    asyncio.run(ann.classification(prompts, model="gemini"))
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

