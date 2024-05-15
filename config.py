VALID_MODELS = ["gemini", "claude"]

GEMINI_CONFIG = {
    "project_config": {
        "qpm": 5,
        "project": "amir-genai-bb", 
        "location": "us-central1"
    },
    "generation_config": {
        "max_output_tokens": 2048,
        "temperature": 0.1,
        "top_p": 1,
    }
}

CLAUDE_CONFIG = {
    "project_config": {
        "qpm": 60,  # Adjust as needed
        "project": "cloud-llm-preview1", 
        "location": "us-central1"
    },
    "generation_config": "" 
}


VERBOSE=False