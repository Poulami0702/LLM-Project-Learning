from langchain.agents import ChainAgent, AgentInput, AgentOutput
from langchain_community.llms import Ollama


# Define a custom agent class to integrate Ollama and tools
class OllamaToolsAgent(ChainAgent):
    print("**** OllamaToolsAgent ****")
    def __init__(self, llm: Ollama, tools: list):
        super().__init__()
        self.llm = llm
        self.tools = tools

    def predict(self, input: AgentInput) -> AgentOutput:
        # Preprocess the input (if necessary)
        processed_input = input.text

        # Generate response using Ollama
        llm_output = self.llm.generate_text(processed_input)

        print("#### self-tools", self.tools)

        print("#### llm_output", llm_output)
        # Apply tools (if any) to the response
        if self.tools:
            for tool in self.tools:
                tool_output = tool(processed_input)  # Use input for tools as well
                llm_output = tool_output + llm_output  # Combine outputs

        return AgentOutput(text=llm_output)