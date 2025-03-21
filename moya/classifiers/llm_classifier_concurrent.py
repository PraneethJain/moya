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

    def classify(
        self,
        message: str,
        thread_id: Optional[str] = None,
        available_agents: List[AgentInfo] = None,
    ) -> List[str]:
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
        select ONLY the agents that are DIRECTLY relevant to handling this specific request. Be HIGHLY selective.

        IMPORTANT GUIDELINES
        1. Choose ONLY agents whose expertise is SPECIFICALLY needed for this exact request
        2. Exclude agents that might be generally useful but aren't needed for this particular task
        3. If the request is about a specific domain, choose ONLY agents with expertise in that domain
        4. Consider the immediate task only - don't select agents for potential follow-up questions
        5. When in doubt, exclude an agent rather than include it
        6. Don't select agents just because their keywords appear in the message
        7. Consider what the user is actually trying to accomplish

        Available agents: {', '.join([f"'{agent.name}: {agent.description}'" for agent in available_agents])}

        User message: {message}

        Return your answer as a comma-separated list of agent IDs ONLY. If no agents are relevant, respond with "none".
        Example outputs:
        - "ec2_agent,security_agent"
        - "logs_agent"
        - "none"
        """

        # Get classification from LLM
        response = self.llm_agent.handle_message(prompt, thread_id=thread_id)

        response_text = response.strip()

        if response_text == "none":
            return [self.default_agent]

        selected_agents_raw = response_text.split(",")
        selected_agents = [agent_name.strip() for agent_name in selected_agents_raw]

        valid_agent_names = [agent.name for agent in available_agents]
        selected_agents = [
            agent for agent in selected_agents if agent in valid_agent_names
        ]

        if not selected_agents:
            return [self.default_agent]

        return selected_agents
