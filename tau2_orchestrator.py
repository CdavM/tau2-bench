"""
Tau2 Orchestrator - Sends config to worker and collects results
Like debate_judge.py in the debate example
"""
import argparse
import uvicorn
from dotenv import load_dotenv
load_dotenv()


from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.a2a.utils.agent_to_a2a import to_a2a
import httpx
from pydantic import BaseModel
from a2a.client import ClientFactory, ClientConfig

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Role,
    TextPart, Part
)

class TauEval(BaseModel):
    average_reward: float
    average_cost: float

def main():
    parser = argparse.ArgumentParser(description="Run tau2 orchestrator")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9100, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    args = parser.parse_args()

    async def send_to_worker(domain: str, agent_llm: str, user_llm: str, num_trials: int, num_tasks: int, worker_url: str = "http://worker:9009") -> str:
        """Send benchmark task to worker agent"""
        print(f"[ORCHESTRATOR TOOL CALLED] send_to_worker: domain={domain}, agent_llm={agent_llm}, user_llm={user_llm}, num_trials={num_trials}, num_tasks={num_tasks}, worker_url={worker_url}")

        try:
            # Create A2A client using ClientFactory with long timeout for benchmarks
            async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0)) as httpx_client:
                client_config = ClientConfig(httpx_client=httpx_client)
                client = await ClientFactory.connect(
                    agent=worker_url,
                    client_config=client_config
                )

                # Construct A2A message using proper types
                import uuid
                message = Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.user,
                    parts=[Part(root=TextPart(
                        text=f"Run benchmark: domain={domain}, agent_llm={agent_llm}, user_llm={user_llm}, num_trials={num_trials}, num_tasks={num_tasks}"
                    ))]
                )

                print(f"[ORCHESTRATOR] Sending A2A message to worker: {message.model_dump()}")

                benchmark_output = ""
                # send_message returns an async generator, collect responses
                async for response in client.send_message(message):
                    print(f"[ORCHESTRATOR] Received response:\n {response}")
                    benchmark_output = str(response)

                print(f"[ORCHESTRATOR] Benchmark completed!")

                # Extract tuple using regex - look for result: followed by tuple
                import re
                match = re.search(r"'result':\s*'(\([\d.]+,\s*[\d.]+\))'", benchmark_output)
                if match:
                    result = match.group(1)
                    print(f"[ORCHESTRATOR] Extracted result: {result}")
                    return result
                else:
                    print(f"[ORCHESTRATOR] Could not extract tuple from output")
                    return "Error: Could not extract result tuple"
        except Exception as e:
            import traceback
            print(f"[ORCHESTRATOR] Error: {e}")
            print(traceback.format_exc())
            return f"Error: {e}"

    root_agent = Agent(
        name="tau2_orchestrator",
        model="gemini-2.5-flash-lite",
        description="Orchestrates tau2 benchmarks across workers",
        instruction="You orchestrate tau2 benchmark execution by sending configurations to worker agents. Use send_to_worker tool. This tool will return a tuple containing average_reward (first number) and average_cost (second number).",
        tools=[FunctionTool(func=send_to_worker)],
        output_schema=TauEval,
    )

    skill = AgentSkill(
        id='orchestrate_benchmark',
        name='Orchestrate Tau2 Benchmark',
        description='Orchestrate tau2 benchmark execution across worker agents',
        tags=['benchmark', 'tau2'],
        examples=['{"worker_url": "http://worker:9200", "domain": "airline", "agent_llm": "gpt-4.1", "user_llm": "gpt-4.1", "num_trials": 1, "num_tasks": 5}']
    )

    agent_card = AgentCard(
        name="tau2_orchestrator",
        description='Orchestrates tau2 benchmarks across workers',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='0.2.1',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()