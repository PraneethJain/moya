from typing import List, Optional

from moya.agents.agent_info import AgentInfo
from moya.classifiers.base_classifier import BaseClassifierConcurrent
from moya.agents.base_agent import Agent


class LLMClassifierConcurrent(BaseClassifierConcurrent):
    """LLM-based classifier for concurrent agent selection."""

    def __init__(self, llm_agent: Agent, default_agent: str):
        """
        Initialize with an LLM agent for classification.
        
        :param llm_agent: An agent that will be used for classification
        :param default_agent: The default agent to use if no specialized match is found
        """
        self.llm_agent = llm_agent
        self.default_agent = default_agent

    def classify(self, message: str, thread_id: Optional[str] = None, available_agents: List[AgentInfo] = None) -> List[str]:
        """
        Use LLM to classify message and select appropriate agent.
        
        :param message: The user message to classify
        :param thread_id: Optional thread ID for context
        :param available_agents: List of available agent names to choose from
        :return: List of relevant agent names (order not significant)
        """
        if not available_agents:
            return None

        # Construct prompt for the LLM
        prompt = f"""Given the following user message and list of available specialized agents,
        select ONLY the agents that are directly relevant to handling this request. Ignore any agents that are not relevant.
        Return only a list of relevant agent ids separated by commas. If no agents are relevant, respond with "none".
        Available agents: {', '.join([f"'{agent.name}: {agent.description}'" for agent in available_agents])}
        User message: {message}
        """

        # Get classification from LLM
        response = self.llm_agent.handle_message(prompt, thread_id=thread_id)

        response_text = response.strip()
        
        if response_text == "none":
            return [self.default_agent]
        
        selected_agents_raw = response_text.split(',')
        selected_agents = [agent_name.strip() for agent_name in selected_agents_raw]
        
        valid_agent_names = [agent.name for agent in available_agents]
        selected_agents = [agent for agent in selected_agents if agent in valid_agent_names]
        
        if not selected_agents:
            return [self.default_agent]
        
        return selected_agents
