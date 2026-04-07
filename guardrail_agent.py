import asyncio
from nemoguardrails import LLMRails, RailsConfig
from rich.console import Console
from rich.panel import Panel
from policy_parser import initialize_rag, get_relevant_policy
import time

console = Console()

class GuardrailAgent:
    def __init__(self, policy_paths=None):
        # 1. Initialize the RAG Vector Database
        if policy_paths:
            initialize_rag(policy_paths)
            
        self.active_policy_chunk = "" 

        # 2. Initialize NeMo Guardrails (NeMo exclusively manages the LLMs now)
        config = RailsConfig.from_path("./config")
        self.rails = LLMRails(config)
        
        # 3. Register the custom action
        self.rails.register_action(self.check_policy_action, name="check_policy_violation")

    # 4. THE FIX: By requesting 'llm' in the signature, NeMo natively injects 
    # its internal, configured model directly into your action.
    async def check_policy_action(self, context: dict, llm):
        """Evaluates the model's output against the RETRIEVED policy chunk."""
        bot_response = context.get("last_bot_message", "")
        
        if not self.active_policy_chunk:
            return True # No policy context found for this query
            
        prompt = f"""
        Company Policy Excerpt:
        {self.active_policy_chunk}
        
        Proposed Agent Response:
        {bot_response}
        
        Does the proposed response violate any rules in the company policy excerpt?
        Answer strictly YES or NO.
        """
        
        # Run the prompt entirely through NeMo's internal pipeline
        check_response = await llm.ainvoke(prompt)
        
        # Parse the response safely
        if "YES" in check_response.content.upper():
            context["last_bot_message"] = "I cannot provide this information as it conflicts with the uploaded security policy."
            return False
            
        return True

    async def chat(self, user_input: str) -> dict:
        t0 = time.time()
        
        # RAG STEP: Search the vector database for the 2 most relevant paragraphs
        self.active_policy_chunk = get_relevant_policy(user_input, k=2)
        
        # Build the message history: System Context + User Query
        messages = []
        if self.active_policy_chunk:
            messages.append({
                "role": "system", 
                "content": f"You are a strict, helpful corporate AI. Use the following policy rules to answer the user safely:\n\n{self.active_policy_chunk}"
            })
            
        messages.append({"role": "user", "content": user_input})
        
        # Run NeMo pipeline
        try:
            response = await self.rails.generate_async(messages=messages)
        except Exception as exc:
            latency = round((time.time() - t0) * 1000)
            return {
                "response": f"Guardrailed model invocation failed: {exc}",
                "latency_ms": latency,
            }
        
        latency = round((time.time() - t0) * 1000)
        final_text = response["role"] == "assistant" and response.get("content") or "Blocked by guardrails."

        return {
            "response": final_text,
            "latency_ms": latency
        }

    async def run(self):
        console.print(Panel(
            "[bold blue]NEMO GUARDRAILED AGENT (RAG ENABLED)[/] \n"
            "[dim]Type 'quit' to exit[/]",
            border_style="blue"
        ))
        while True:
            user_in = input("\n[guarded] You: ").strip()
            if user_in.lower() == "quit": break
            if not user_in: continue

            result = await self.chat(user_in)
            console.print(
                f"\n[bold blue]Agent:[/] {result['response']}"
                f"\n[dim]⏱ {result['latency_ms']}ms[/]"
            )

if __name__ == "__main__":
    # Add one or more PDF paths here.
    agent = GuardrailAgent(policy_paths=[
        r"C:\Users\c-mzaki\Documents\ai-agent-lab\Indian Company Policies and NDA.pdf",
        r"C:\Users\c-mzaki\Documents\ai-agent-lab\Company Policy  NDA.pdf"
    ])
    asyncio.run(agent.run())