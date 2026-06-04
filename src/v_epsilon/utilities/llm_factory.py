import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: str, temperature: float = 0.0, **kwargs):
    """
    Returns the appropriate LangChain LLM instance based on model string.

    Examples:
        get_llm("gpt-4.1")
        get_llm("claude-4-sonnet")
        get_llm("gemini-2.5-pro")
        get_llm("llama-4-scout")
        get_llm("ollama:llama3")
    """

    model = model_name.lower()

    # OpenAI
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )

    # Anthropic
    if model.startswith("claude"):
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    # Google Gemini
    if model.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            **kwargs,
        )

    # Groq-hosted models
    if model.startswith(
        (
            "llama",
            "mixtral",
            "deepseek",
            "qwen",
        )
    ):
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            **kwargs,
        )

    # Local Ollama
    if model.startswith("ollama:"):
        ollama_model = model_name.split(":", 1)[1]
        return ChatOllama(
            model=ollama_model,
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(f"Unsupported model: {model_name}")
