import time
import streamlit as st

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

options = st.multiselect(
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


We are using a simple template for all models (remember it's about ensemble of weak classifiers)

```
<CONTEXT>
TBD
</CONTEXT>
------------

<ANSWERs>
TBD
</ANSWERs>
------------

INSTRUCTION:
- read the above context carefully.
- take your time and pick the precise correct answer from <ANSWERS> for the given <CONTEXT>.
- return the exact correct answer from <ANSWERS>. Don't provide explanations.
```


Add your instructions as bullet points in the following box.

"""
)

insturctions = st.text_input("""

""")

st.markdown("""---""")

def generate_response():
    # TODO: map this to utils function 
    time.sleep(2)

if st.button("🤖 Initiate LLMS!"):
    generate_response()
