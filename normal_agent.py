from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from policy_parser import initialize_rag, get_relevant_policy
import time
import requests

console = Console()

class NormalAgent:
    def __init__(self, policy_paths=None):
        # 1. Initialize the RAG Vector Database
        if policy_paths:
            initialize_rag(policy_paths)

        # 2. Use the stable OpenAI wrapper pointing to local Ollama
        self.llm = ChatOpenAI(
            model="mistral:7b", # Or gemma2:9b if you switched to it!
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            temperature=0.7
        )
        self.history = []

    def chat(self, user_input: str) -> dict:
        t0 = time.time()

        # Step 1: RAG Retrieval - get relevant paragraphs
        policy_chunk = get_relevant_policy(user_input, k=2)

        # Step 2: Build the system prompt dynamically (Zero Guardrails)
        system_prompt = "You are a helpful AI assistant. Answer the user's questions directly."
        if policy_chunk:
            system_prompt += f"\n\nContext to help answer the user:\n{policy_chunk}"

        # Combine system prompt + history + new user input
        current_messages = [SystemMessage(content=system_prompt)] + self.history + [HumanMessage(content=user_input)]

        # Step 3: Invoke the model directly (No safety middleware)
        try:
            response = self.llm.invoke(current_messages)
        except requests.exceptions.RequestException:
            latency = round((time.time() - t0) * 1000)
            return {
                "response": "I couldn't reach Ollama at http://127.0.0.1:11434. Start Ollama (for example: 'ollama serve') and try again.",
                "latency_ms": latency,
            }
        except Exception as exc:
            latency = round((time.time() - t0) * 1000)
            return {
                "response": f"Model invocation failed: {exc}",
                "latency_ms": latency,
            }

        latency = round((time.time() - t0) * 1000)

        # Step 4: Save raw history
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response.content))

        return {
            "response": response.content,
            "latency_ms": latency
        }

    def run(self):
        console.print(Panel(
            "[bold red]BARE AGENT (RAG ENABLED)[/] — No guardrails active\n"
            "[dim]Type 'quit' to exit[/]",
            border_style="red"
        ))
        while True:
            user_in = input("\n[bare] You: ").strip()
            if user_in.lower() == "quit": break
            if not user_in: continue

            result = self.chat(user_in)
            console.print(
                f"\n[bold red]Agent:[/] {result['response']}"
                f"\n[dim]⏱ {result['latency_ms']}ms[/]"
            )

if __name__ == "__main__":
    # Add one or more PDF paths here.
    agent = NormalAgent(policy_paths=[
        r"C:\Users\c-mzaki\Documents\ai-agent-lab\Indian Company Policies and NDA.pdf",
        r"C:\Users\c-mzaki\Documents\ai-agent-lab\Company Policy  NDA.pdf"
    ])
    agent.run()