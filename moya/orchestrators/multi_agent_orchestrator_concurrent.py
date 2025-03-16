import threading
from typing import Optional
from moya.orchestrators.base_orchestrator import BaseOrchestrator
from moya.registry.agent_registry import AgentRegistry
from moya.classifiers.base_classifier import BaseClassifierConcurrent
from moya.tools.ephemeral_memory import EphemeralMemory


class MultiAgentOrchestratorConcurrent(BaseOrchestrator):
    """
    A concurrent orchestrator that uses a classifier to route messages to appropriate agents.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        classifier: BaseClassifierConcurrent,
        default_agent_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """
        :param agent_registry: The AgentRegistry to retrieve agents from
        :param classifier: The classifier to use for agent selection
        :param default_agent_name: Fallback agent if classification fails
        :param config: Optional configuration dictionary
        """
        super().__init__(agent_registry=agent_registry, config=config)
        self.classifier = classifier
        self.default_agent_name = default_agent_name

    def orchestrate(
        self, thread_id: str, user_message: str, stream_callback=None, **kwargs
    ) -> str:
        """
        Orchestrate the message handling using intelligent agent selection.

        :param thread_id: The conversation thread ID
        :param user_message: The message from the user
        :param stream_callback: Optional callback for streaming responses
        :param kwargs: Additional context
        :return: The concatenated response from all the chosen agents
        """
        EphemeralMemory.store_message(
            thread_id=thread_id, sender="user", content=user_message
        )

        all_responses = []
        current_message = user_message
        max_iterations = 20
        available_agents = self.agent_registry.list_agents()
        if not available_agents:
            return "[No agents available to handle message.]"

        for _ in range(max_iterations):

            agent_names = self.classifier.classify(
                message=current_message,
                thread_id=thread_id,
                available_agents=available_agents,
            )

            if not agent_names and self.default_agent_name:
                agent_names = [self.default_agent_name]

            agents = [
                self.agent_registry.get_agent(name) for name in agent_names if name
            ]
            agents = [agent for agent in agents if agent]

            if not agents:
                if all_responses:
                    return (
                        "\n\n".join(all_responses)
                        + "\n\n[No suitable agent found for next step.]"
                    )
                return "[No suitable agent found to handle message.]"

            responses = {}

            def run_agent(agent):
                agent_prefix = f"[{agent.agent_name}] "
                agent_response = agent.handle_message(
                    current_message, thread_id=thread_id, **kwargs
                )
                responses[agent.agent_name] = agent_prefix + agent_response

            threads = []
            for agent in agents:
                thread = threading.Thread(target=run_agent, args=(agent,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            current_output = "\n\n".join(responses.values())
            all_responses.append(current_output)

            if "STOP" in current_output:
                if stream_callback:
                    stream_callback("\n[Workflow stopped based on agent decision]\n")
                break

            if "NEXT_STEP" in current_output or "CONTINUE" in current_output:
                if "NEXT_MESSAGE:" in current_output:
                    parts = current_output.split("NEXT_MESSAGE:")
                    if len(parts) > 1:
                        current_message = parts[1].split("\n")[0].strip()
                    else:
                        current_message = f"Based on these results, what action should be taken?\n{current_output}"
                else:
                    current_message = (
                        f"Continue processing based on these results:\n{current_output}"
                    )

                if stream_callback:
                    stream_callback("\n[Processing next step...]\n")
            else:
                break

        final_response = "\n\n".join(all_responses)

        EphemeralMemory.store_message(
            thread_id=thread_id,
            sender="MultiAgentOrchestratorConcurrent",
            content=final_response,
        )

        return final_response
