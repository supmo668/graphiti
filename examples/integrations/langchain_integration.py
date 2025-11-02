"""
LangChain Integration Example

This example demonstrates how to use Graphiti as a memory backend for LangChain agents,
providing persistent, temporally-aware memory that evolves with agent interactions.

Use Case:
- Multi-turn conversations with persistent context
- Knowledge accumulation across sessions
- Temporal reasoning about past interactions
- Graph-based context retrieval

Requirements:
- langchain
- langchain-openai
- graphiti-core

Research Reference:
- Zep: Temporal Knowledge Graph Architecture https://arxiv.org/abs/2501.13956
- LangChain Memory: https://python.langchain.com/docs/modules/memory/
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

# Load environment variables
load_dotenv()


class GraphitiMemory:
    """
    Wrapper for Graphiti to provide LangChain-compatible memory interface.
    
    This class bridges Graphiti's graph-based memory with LangChain's memory protocol,
    allowing agents to store and retrieve context from a knowledge graph.
    """

    def __init__(self, graphiti: Graphiti, group_id: str = 'langchain-agent'):
        self.graphiti = graphiti
        self.group_id = group_id
        self.conversation_history = []

    async def add_message(self, role: str, content: str):
        """Add a message to Graphiti's episodic memory."""
        timestamp = datetime.now()
        
        # Store as episodic node
        episode_text = f"{role}: {content}"
        await self.graphiti.add_episode(
            name=f"conversation_{timestamp.isoformat()}",
            episode_body=episode_text,
            source_description=f"LangChain {role} message",
            group_id=self.group_id,
        )
        
        self.conversation_history.append({'role': role, 'content': content})

    async def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """Retrieve relevant context from graph for a given query."""
        # Search for relevant episodes and entities
        results = await self.graphiti.search(query, group_ids=[self.group_id])
        
        # Format context from search results
        context_parts = []
        
        # Add relevant nodes
        for node in results.nodes[:limit]:
            if isinstance(node, EntityNode):
                context_parts.append(f"Entity: {node.name} - {node.summary}")
            elif isinstance(node, EpisodicNode):
                context_parts.append(f"Memory: {node.content}")
        
        # Add relevant edges (relationships)
        for edge in results.edges[:limit]:
            if isinstance(edge, EntityEdge):
                context_parts.append(
                    f"Relationship: {edge.source_node_uuid} {edge.name} {edge.target_node_uuid}"
                )
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."

    def clear(self):
        """Clear conversation history (graph persists)."""
        self.conversation_history = []


async def main():
    """
    Demonstrate LangChain agent with Graphiti memory.
    
    This example shows:
    1. Setting up Graphiti as a memory backend
    2. Running a multi-turn conversation
    3. Retrieving context from the knowledge graph
    4. Using temporal context for better responses
    """
    
    # Initialize Graphiti
    print("Initializing Graphiti...")
    graphiti = Graphiti(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    await graphiti.build_indices_and_constraints()
    
    # Initialize memory wrapper
    memory = GraphitiMemory(graphiti, group_id="langchain-example")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    print("\n" + "="*60)
    print("LangChain + Graphiti Integration Example")
    print("="*60)
    
    # Example conversation
    conversations = [
        ("user", "My name is Alice and I work as a software engineer at TechCorp."),
        ("user", "I'm working on a knowledge graph project using Python."),
        ("user", "What do you know about me?"),
    ]
    
    for role, message in conversations:
        print(f"\n{role.upper()}: {message}")
        
        # Add message to Graphiti
        await memory.add_message(role, message)
        
        # For user queries, retrieve context and generate response
        if role == "user":
            # Get relevant context from graph
            context = await memory.get_relevant_context(message)
            print(f"\nRetrieved Context:\n{context}")
            
            # Generate response with context
            prompt = f"""Based on the following context from our conversation history:

{context}

User query: {message}

Please provide a helpful response based on what you know from the context."""
            
            response = llm.predict(prompt)
            print(f"\nASSISTANT: {response}")
            
            # Store assistant response
            await memory.add_message("assistant", response)
    
    # Demonstrate advanced search
    print("\n" + "="*60)
    print("Advanced Graph Search")
    print("="*60)
    
    search_query = "software engineer"
    print(f"\nSearching graph for: '{search_query}'")
    
    results = await graphiti.search(search_query, group_ids=["langchain-example"])
    
    print(f"\nFound {len(results.nodes)} nodes and {len(results.edges)} edges")
    for node in results.nodes[:3]:
        print(f"  - {node.name}: {node.summary if hasattr(node, 'summary') else 'N/A'}")
    
    # Cleanup
    await graphiti.close()
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
