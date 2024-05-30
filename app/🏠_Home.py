import sys
import json
import time
import streamlit as st

import pandas as pd
from collections import Counter
from typing import Dict, List
from sklearn.metrics import accuracy_score

sys.path.insert(1, "/usr/local/google/home/amirimani/Desktop/projects/llm_annotator")
from config import PALM_CONFIG, GEMINI_CONFIG

st.title('LLM Ensemble - Whose Vote to Count?')

col1, col2 = st.columns(2)
user_data = col1.file_uploader("Upload your own data")
benchmark_data = col2.selectbox(
   "Popular LLM Benchmarks",
   ("ARC", "HellaSwag", "HumanEval", "MMLU", "SuperGLUE"),
   index=None,
   placeholder="Select a dataset...",
)
st.write("You selected:", benchmark_data)

model_options = st.multiselect(
    "Select your models from Vertex Model Garden",
    [
     "Gemini 1.0 Pro", "Gemini 1.5 Pro", "Gemini 1.5 Flash", 
     "Gemini Ultra", "Gemma 2b", "Gemma 7b", "Claude 3 Haiku", 
     "Claude 3 Sonnet", "PaLM 2 Text Bison"
     ],
    ["Gemini 1.0 Pro", "Gemini 1.5 Pro"]
    )


st.markdown(
    f"""
#### Prompt Template


For the MMLU example, we are using the below prompt for all models (remember it's about ensemble of "weak" classifiers not the most accurate models)

```
<QUESTION>
mmlu_question
</QUESTION>
------------

<CHOICES>
mmlu_choices
</choices>
------------

INSTRUCTION:
- read the above question carefully.
- you are given 4 choices seperated by comma in <CHOICES>.
- take your time and pick the precise correct answer from <CHOICES> for the given <QUESTION>.
- remember that there is always only one correct answer.
- return the exact correct answer from <CHOICES>. Don't provide explanations.
```
"""
)

# insturctions = st.text_area("""
# Your own prompt here. Don't over engineer!
# """)

st.markdown("""---""")

def generate_response():

    seed = 42
    n_sample = 100
    with st.spinner("💾 Loading data..."):
        from datasets import load_dataset
        dataset = load_dataset("cais/mmlu", "all")
        n_sample = 100
        # # take a small sample for dev purposes
        dataset = dataset['test'].shuffle(seed=seed).select(range(n_sample))
    
    gemini_prompt_template = """
        <QUESTION>
        {datapoint}
        </QUESTION>
        ------------

        <CHOICES>
        {labels}
        </choices>
        ------------

        INSTRUCTION:
        - read the above question carefully.
        - you are given 4 choices seperated by comma in <CHOICES>.
        - take your time and pick the precise correct answer from <CHOICES> for the given <QUESTION>.
        - remember that there is always only one correct answer.
        - return the exact correct answer from <CHOICES>. Don't provide explanations.
    """
    prompt = [gemini_prompt_template.format(datapoint=x['question'],
                                        labels=x['choices']) for x in dataset]
    # st.write(model_options)

    progress_text = " 🤖 Generating LLM responses. Please wait..."
    with st.spinner(progress_text):
        time.sleep(5)

    return dataset


def get_majority_vote(label_dict: Dict[str, List[int]]) -> List[int]:
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


def accuracy_with_none_penalty(y_true, y_pred):
    filtered_y_true = []
    filtered_y_pred = []

    for true, pred in zip(y_true, y_pred):
        if pred is not None:  # Only include non-None predictions
            filtered_y_true.append(true)
            filtered_y_pred.append(pred)
        else:
            filtered_y_true.append(true)  # Include true label
            filtered_y_pred.append(-1)   # Replace None with wrong label (e.g., -1)

    return accuracy_score(filtered_y_true, filtered_y_pred)


    



if st.button("🤖 Initiate LLMS!"):
    # dataset = generate_response()
    progress_text = "All LLM responses are collected. Now aggregating results using GLAD algorithm ⚡"
    my_bar = st.progress(0, text=progress_text)

    for precent_complete in range(100):
        time.sleep(0.01)
        from utils import glad
        
    time.sleep(0.08)
    my_bar.progress(precent_complete + 1, text=progress_text)
    glad_output = glad("./data/20240530/llm_response_50__20240530.txt")
    with open("./data/20240530/llm_response_50__20240530.json", "r") as f:
        llm_response = json.load(f)

    df_gt = pd.read_csv('./data/20240530/gt_50__20240530.csv')


    with st.spinner("🚧 Calculating Majority Vote and accuracy metrics"):
        llm_response["majority"] = get_majority_vote(llm_response)
        llm_response["glad"] = list(glad_output['labels'].values())
        d = {}
        for k, v in llm_response.items():
            acc_value = round(accuracy_with_none_penalty(list(df_gt['gt'].values), v) * 100, 2)
            d[k] = f"{acc_value} %"

        df_acc = pd.DataFrame([d]).T.reset_index()
        df.columns = ["model", "accuracy"]
        # styler = df.style.format(subset=['accuracy'], decimal=',', precision=1)
        # st.write(styler.to_html(), unsafe_allow_html=True)
        st.dataframe(df_acc)


    # data explorer
    question_d = {"question": df_gt['question'],
                "task difficulty": list(glad_output["beta"].values()),
                "label confidence": list(glad_output["probZ"].values())
                }
    
    df_question = pd.DataFrame.from_dict(question_d)
    

    time.sleep(2)
    st.markdown("""---""")


    st.dataframe(df_question)
