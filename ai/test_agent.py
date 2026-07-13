from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)

response = llm.invoke(
    "Sen QuantForge projesinde çalışan kıdemli Python ve AI trading mühendisisin. Kendini tanıt."
)

print(response.content)