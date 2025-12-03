"""
Tau2 Worker Agent - Receives benchmark config and executes it
Like debater.py in the debate example
"""
import argparse
import subprocess
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

def main():
    parser = argparse.ArgumentParser(description="Run tau2 worker agent")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9200, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    args = parser.parse_args()

    def run_tau2_benchmark(
        domain: str,
        agent_llm: str,
        user_llm: str,
        num_trials: int = 1,
        num_tasks: int = 5
    ) -> str:
        """Execute tau2 benchmark with given parameters"""
        print(f"[WORKER TOOL CALLED] run_tau2_benchmark: domain={domain}, agent_llm={agent_llm}, user_llm={user_llm}, num_trials={num_trials}, num_tasks={num_tasks}")

        cmd = [
            "tau2", "run",
            "--domain", domain,
            "--agent-llm", agent_llm,
            "--user-llm", user_llm,
            "--num-trials", str(num_trials),
            "--num-tasks", str(num_tasks)
        ]

        print(f"[WORKER] Executing command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print(f"[WORKER] Command finished with exit code: {result.returncode}")
            print(f"[WORKER] Stdout length: {len(result.stdout)} chars")
            print(f"[WORKER] Stderr length: {len(result.stderr)} chars")

            if result.returncode == 0:
                # Extract Average Reward and Average Cost from stdout
                import re
                reward_match = re.search(r'Average Reward:\s*([\d.]+)', result.stdout)
                cost_match = re.search(r'Average Cost per Conversation:\s*\$([\d.]+)', result.stdout)

                if reward_match and cost_match:
                    avg_reward = float(reward_match.group(1))
                    avg_cost = float(cost_match.group(1))
                    output = f"({avg_reward}, {avg_cost})"
                    print(f"[WORKER] Extracted metrics: {output}")
                    return output
                else:
                    print(f"[WORKER] Could not extract metrics from output")
                    return "Error: Could not extract metrics from benchmark output"
            else:
                output = f"❌ Benchmark failed with exit code {result.returncode}"
                print(f"[WORKER] Returning error response")
                return output
        except subprocess.TimeoutExpired:
            print("[WORKER] Command timed out")
            return "❌ Error: Benchmark execution timed out after 1 hour"
        except Exception as e:
            print(f"[WORKER] Error executing command: {e}")
            return f"❌ Error executing benchmark: {str(e)}"

    root_agent = Agent(
        name="tau2_worker",
        model="gemini-2.5-flash-lite",
        description="Executes tau2 benchmarks",
        instruction="When you receive a benchmark request: 1) Call the run_tau2_benchmark tool with the provided parameters, 2) Process the summary output and return a pair (Average Reward, Average Cost per Conversation) to the agent who requested the benchmarks, 3) Stop and finish - do not ask follow-up questions or retry.",
        tools=[run_tau2_benchmark]
    )

    agent_card = AgentCard(
        name="tau2_worker",
        description='Executes tau2 benchmarks',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='0.2.1',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()