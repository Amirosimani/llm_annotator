# LLM Annotators

Creating and maintaining high-quality training data is essential for unlocking the full potential of generative AI models. It not only serves as the foundational knowledge base during initial training but also plays a crucial role in subsequent fine-tuning and grounding processes.

* Available models: Gemini, Claude, PaLM, Gemma
* Available modality: Text, Image
* Available tasks: Classfication (majority vote, GLAD <sup>1</sup> )

## Run the Streamlit app locally
- create a virutal env `python3 -m venv .venv`
- activate the env `source .venv/bin/activate`
- install requirments `pip install -r requirments.txt`
- **local secret management**: `.streamlit/secrets.toml` for local runs. make sure you add it to `.gitignore`
- run the app:
 - `streamlit run ./app/🏠_Home.py`




# References
1. [GLAD paper](https://proceedings.neurips.cc/paper_files/paper/2009/file/f899139df5e1059396431415e770c6dd-Paper.pdf) and implementation [referece](https://github.com/notani/python-glad/blob/master/glad.py#L58)