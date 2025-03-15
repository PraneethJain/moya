from moya.agents.azure_openai_agent import AzureOpenAIAgent, AzureOpenAIAgentConfig
from moya.tools.base_tool import BaseTool

from new_func import generate_tool


class AzureOpenAIDynamicToolingAgent(AzureOpenAIAgent):
    def __init__(self, config: AzureOpenAIAgentConfig):
        super().__init__(config=config)
        self.generate_dynamic_tool_tool = BaseTool(name="generate_dynamic_tool_tool",
                                                   description="Tool to generate a new tool dynamically and add it to this agent",
                                                   function=self.generate_dynamic_tool_fn, parameters={
                "requirement": {"type": "string",
                                "description": "The description of the task to be achieved by the newly created tool"}},
                                                   required=["text"])
        self.tool_registry.register_tool(self.generate_dynamic_tool_tool)
        self.system_prompt += "\nHowever, if you think that none of the tools that you have can perform the task requested by the user, then default to generate_dynamic_tool_tool which you have access to."

    def generate_dynamic_tool_fn(self, requirement: str):
        tools = [{"name": tool.name, "description": tool.description, "parameters": {
                        name: {
                            "type": info["type"],
                            "description": info["description"]
                        } for name, info in tool.parameters.items()
                }} for tool in self.tool_registry.get_tools()]

        # Responsible for generating a tool dynamically
        result = generate_tool(query=requirement, agent_name=self.agent_name, agent_description=self.description,
                               tools=tools)

        # Get generated code
        code = result["code"]
        new_tool_name = result["function_name"]

        # Add this tool to the agent
        exec(code)
        self.tool_registry.register_tool(new_tool_name)

        # Update the system prompt with the new tool info
        # updated_tool_prompt += f"\nYou have access to the {new_tool_name} tool which does {tool_description}"
