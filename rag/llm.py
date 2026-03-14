# rag/llm.py

from langchain_ibm import WatsonxLLM
from langchain.llms import GPT4All, HuggingFaceHub

def get_llm(model_name="mixtral", project_id="skills-network", params=None):
    """
    Returns a LangChain LLM based on user selection.
    
    Supports free models like:
        - mixtral: Watsonx hosted (default)
        - gpt4all: Local lightweight
        - bge-small: HuggingFace hosted BigScience
        - oasst: OpenAssistant hosted model
        - mpt-7b: HuggingFace instruction-tuned
        - vicuna-7b: HuggingFace instruction-tuned
    """

    # Default parameters (output length/ randomness)
    if params is None:
        params = {"max_new_tokens": 256, "temperature": 0.5}

    model_name = model_name.lower()

    # Mixtral-8x7B Instruct v01
    if model_name == "mixtral":
        return WatsonxLLM(
            model_id="mistralai/mixtral-8x7b-instruct-v01",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=project_id,
            params=params,
        )

    # GPT4All-Lora Quantized - latest version
    elif model_name == "gpt4all":
        return GPT4All(model="gpt4all-lora-quantized")

    # BigScience Big Generative Encoder (BGE) Small
    elif model_name == "bge-small":
        return HuggingFaceHub(repo_id="bigscience/bge-small", model_kwargs=params)

    # OpenAssistant SFT-4 Pythia 12B
    elif model_name == "oasst":
        return HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b", model_kwargs=params)

    # MPT-7B Instruct (MosaicML)
    elif model_name == "mpt-7b":
        return HuggingFaceHub(repo_id="mosaicml/mpt-7b-instruct", model_kwargs=params)

    # Vicuna 7B v1.3
    elif model_name == "vicuna-7b":
        return HuggingFaceHub(repo_id="vicuna/7b-v1.3", model_kwargs=params)

    else:
        raise ValueError(f"The LLM '{model_name}' not supported. Available models: mixtral, gpt4all, bge-small, oasst, mpt-7b, vicuna-7b")