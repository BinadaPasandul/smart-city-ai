"""Interactive CLI test script for testing the Mobility Agent via the Orchestrator."""

import asyncio
import sys
from pathlib import Path

# Ensure backend directory is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.bootstrap import create_agent_registry, create_orchestrator
from app.agents.contracts import AgentRequest


async def main():
    print("==================================================")
    print("      Smart City AI — Mobility Agent Tester       ")
    print("==================================================")
    print("Initializing Agent Registry & City Orchestrator...")
    
    registry = create_agent_registry()
    orchestrator = create_orchestrator(registry)
    
    sample_queries = [
        "Where can I park near Colombo Fort?",
        "What's the traffic situation on Galle Road?",
        "Find EV charging stations in Kollupitiya with CCS2 plug",
        "What is the bus schedule for route 138?",
        "How do I travel from Colombo Fort to Kandy?",
    ]
    
    print("\nSelect a sample query or type your own:")
    for idx, q in enumerate(sample_queries, 1):
        print(f"  [{idx}] {q}")
    print("  [0] Type custom query")
    
    try:
        choice = input("\nEnter choice (0-5) [default: 1]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if choice in ("1", "2", "3", "4", "5"):
        query = sample_queries[int(choice) - 1]
    elif choice == "0":
        query = input("Enter your custom query: ").strip()
        if not query:
            print("Query cannot be empty.")
            return
    else:
        query = sample_queries[0]

    print(f"\n--- Submitting Query: '{query}' ---")
    request = AgentRequest(query=query)
    response = await orchestrator.execute(request)

    print("\n--- Response ---")
    print(f"Success: {response.success}")
    print(f"Executing Agent(s): {response.metadata.get('selected_agents')}")
    print(f"Routing Method: {response.metadata.get('routing_method')}")
    print("\nAnswer:")
    print(response.answer)
    print("\nAttributed Sources:")
    for s in response.sources:
        print(f" - [{s.source_type}] {s.name} (Location: {s.metadata.get('location')})")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
