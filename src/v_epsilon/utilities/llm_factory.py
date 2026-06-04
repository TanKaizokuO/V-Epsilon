from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama


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
            **kwargs,
        )

    # Anthropic
    if model.startswith("claude"):
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            **kwargs,
        )

    # Google Gemini
    if model.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
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
