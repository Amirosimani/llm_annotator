import re
import logging
import asyncio
import numpy as np
import scipy as sp
from datetime import datetime
from collections import Counter
from itertools import combinations
from asynciolimiter import Limiter
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Optional


import vertexai


from config import VALID_MODELS, GEMINI_CONFIG, PALM_CONFIG, VERBOSE


def init_logger():
    global logger
    logger = logging.getLogger('base')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

init_logger()


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



THRESHOLD = 1e-5

verbose = False
debug = False
logger = None

class Dataset(object):
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


def load_data(filename):
    data = Dataset()
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
        if verbose:
            logger.info('Reading {} labels of {} labelers over {} tasks for prior P(Z) = {}'.format(data.numLabels,
                                                                                                    data.numLabelers,
                                                                                                    data.numTasks,
                                                                                                    data.priorZ))
        # Read Labels
        data.labels = np.zeros((data.numTasks, data.numLabelers))
        for line in f:
            task, labeler, label = map(int, line.split())
            if debug:
                logger.info("Read: task({})={} by labeler {}".format(task, label, labeler))
            data.labels[task][labeler] = label + 1
    # Initialize Probs
    data.priorAlpha = np.ones(data.numLabelers)
    data.priorBeta = np.ones(data.numTasks)
    data.probZ = np.empty((data.numTasks, data.numClasses))
    # data.priorZ = (np.zeros((data.numClasses, data.numTasks)).T + data.priorZ).T
    data.beta = np.empty(data.numTasks)
    data.alpha = np.empty(data.numLabelers)

    return data


def init_logger():
    global logger
    logger = logging.getLogger('GLAD')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logsigmoid(x):
    return - np.log(1 + np.exp(-x))

def EM(data):
    u"""Infer true labels, tasks' difficulty and workers' ability
    """
    # Initialize parameters to starting values
    data.alpha = data.priorAlpha.copy()
    data.beta = data.priorBeta.copy()
    data.probZ[:] = data.priorZ[:]

    EStep(data)
    lastQ = computeQ(data)
    MStep(data)
    Q = computeQ(data)
    counter = 1
    while abs((Q - lastQ) / lastQ) > THRESHOLD:
        if verbose: logger.info('EM: iter={}'.format(counter))
        lastQ = Q
        EStep(data)
        MStep(data)
        Q = computeQ(data)
        counter += 1


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
        correct = logsigmoid(exponents[delta]).sum()
        # Log likelihood for the observations s.t. l_ij != z_j
        wrong = (logsigmoid(-exponents[oneMinusDelta]) - np.log(float(data.numClasses - 1))).sum()

        # Return log likelihood
        return correct + wrong

    if verbose: logger.info('EStep')
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


def packX(data):
    return np.r_[data.alpha.copy(), data.beta.copy()]


def unpackX(x, data):
    data.alpha = x[:data.numLabelers].copy()
    data.beta = x[data.numLabelers:].copy()


def getBoundsX(data, alpha=(-100, 100), beta=(-100, 100)):
    alpha_bounds = np.array([[alpha[0], alpha[1]] for i in range(data.numLabelers)])
    beta_bounds = np.array([[beta[0], beta[1]] for i in range(data.numLabelers)])
    return np.r_[alpha_bounds, beta_bounds]


def f(x, *args):
    u"""Return the value of the objective function
    """
    data = args[0]
    d = Dataset(labels=data.labels, numLabels=data.numLabels, numLabelers=data.numLabelers,
                numTasks=data.numTasks, numClasses=data.numClasses,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta,
                priorZ=data.priorZ, probZ=data.probZ)
    unpackX(x, d)
    return - computeQ(d)


def df(x, *args):
    u"""Return gradient vector
    """
    data = args[0]
    d = Dataset(labels=data.labels, numLabels=data.numLabels, numLabelers=data.numLabelers,
                numTasks=data.numTasks, numClasses=data.numClasses,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta,
                priorZ=data.priorZ, probZ=data.probZ)
    unpackX(x, d)
    dQdAlpha, dQdBeta = gradientQ(d)
    # Flip the sign since we want to minimize
    assert not np.any(np.isinf(dQdAlpha)), 'Invalid Gradient Value [Alpha]'
    assert not np.any(np.isinf(dQdBeta)), 'Invalid Gradient Value [Beta]'
    assert not np.any(np.isnan(dQdAlpha)), 'Invalid Gradient Value [Alpha]'
    assert not np.any(np.isnan(dQdBeta)), 'Invalid Gradient Value [Beta]'
    return np.r_[-dQdAlpha, -dQdBeta]


def MStep(data):
    if verbose: logger.info('MStep')
    initial_params = packX(data)
    params = sp.optimize.minimize(fun=f, x0=initial_params, args=(data,), method='CG',
                                  jac=df, tol=0.01,
                                  options={'maxiter': 25, 'disp': verbose})
    if debug:
        logger.debug(params)
    unpackX(params.x, data)


def computeQ(data):
    u"""Calculate the expectation of the joint likelihood
    """
    Q = 0
    # Start with the expectation of the sum of priors over all tasks
    Q += (data.probZ * np.log(data.priorZ)).sum()

    # the expectation of the sum of posteriors over all tasks
    ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))

    # logSigma = - np.log(1 + np.exp(-ab))
    logSigma = logsigmoid(ab)  # logP
    idxna = np.isnan(logSigma)
    if np.any(idxna):
        logger.warning('an invalid value was assigned to np.log [computeQ]')
        logSigma[idxna] = ab[idxna]  # For large negative x, -log(1 + exp(-x)) = x

    # logOneMinusSigma = - np.log(1 + np.exp(ab))
    logOneMinusSigma = logsigmoid(-ab) - np.log(float(data.numClasses - 1))  # log((1-P)/(K-1))
    idxna = np.isnan(logOneMinusSigma)
    if np.any(idxna):
        logger.warning('an invalid value was assigned to np.log [computeQ]')
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

    if debug:
        logger.debug('a[0]={} a[1]={} a[2]={} b[0]={}'.format(data.alpha[0], data.alpha[1],
                                                              data.alpha[2], data.beta[0]))
        logger.debug('Q={}'.format(Q))
    if np.isnan(Q):
        return -np.inf
    return Q


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

    sigma = sigmoid(ab)
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

    if debug:
        logger.debug('dQdAlpha[0]={} dQdAlpha[1]={} dQdAlpha[2]={} dQdBeta[0]={}'.format(dQdAlpha[0], dQdAlpha[1],
                                                                                         dQdAlpha[2], dQdBeta[0]))
    return dQdAlpha, dQdBeta


def output(data, save=True):
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
            alpha = np.c_[np.arange(data.numLabelers), data.alpha]
            np.savetxt(f'data/alpha__{now}.csv', alpha, fmt=['%d', '%.5f'], delimiter=',', header='id,alpha')
            beta = np.c_[np.arange(data.numTasks), np.exp(data.beta)]
            np.savetxt(f'data/beta__{now}.csv', beta, fmt=['%d', '%.5f'], delimiter=',', header='id,beta')
            probZ = np.c_[np.arange(data.numTasks), data.probZ]
            np.savetxt(fname=f'data/probZ__{now}.csv',
                    X=probZ,
                    fmt=['%d'] + (['%.5f'] * data.numClasses),
                    delimiter=',',
                    header='id,' + ','.join(['z' + str(k) for k in range(data.numClasses)]))
            label = np.c_[np.arange(data.numTasks), np.argmax(data.probZ, axis=1)]
            np.savetxt(f'data/label_glad__{now}.csv', label, fmt=['%d', '%d'], delimiter=',', header='id,label')

        return results


def glad(filename):
    # data = load_data(data, task_config)
    data = load_data(filename)
    EM(data)

    return output(data)



if __name__ == "__main__":
    ann = Annotate(verbose=VERBOSE)
    import time
    s = time.perf_counter()
    prompts = ["where is paris?", "whare is tehran"]
    asyncio.run(ann.classification(prompts, model="gemini"))
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")


