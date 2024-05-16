from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
# prompt = hub.pull("smithing-gold/assumption-checker")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
print(prompt)