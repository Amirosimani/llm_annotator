VALID_MODELS = ["gemini", "claude", "palm"]

VERBOSE=False

GEMINI_CONFIG = {
    "config_name": "default",
    "model": "gemini-1.0-pro-002",
    "project_config": {
        "qpm": 300,
        "project": "amir-genai-bb", 
        "location": "us-central1"
    },
    "generation_config": {
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 1,
    }
}

PALM_CONFIG = {
    "config_name": "default",
    "model": "text-bison",
    "project_config": {
        "qpm": 300,
        "project": "amir-genai-bb", 
        "location": "us-central1"
    },
    "generation_config": {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.4,
        "top_p": 1
        }
}

CLAUDE_CONFIG = {
    "config_name": "default",
    "model":"claude-3-haiku@20240307",
    "project_config": {
        "qpm": 60,  # Adjust as needed
        "project": "amir-genai-bb", 
        "location": "us-central1"
    },
    "generation_config": "" 
}


