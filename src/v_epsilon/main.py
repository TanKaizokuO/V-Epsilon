from v_epsilon.utilities.llm_factory import get_llm

print(
    "Initializing LLM... ",
)
print(get_llm("gemini-2.5-flash").invoke("Hello, gemini"))
