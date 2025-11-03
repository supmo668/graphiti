"""
LangChain Integration - Graphiti Memory Backend

Demonstrates using Graphiti as a memory backend for LangChain agents.
See README.md for setup instructions.
"""

import asyncio
import os
from datetime import datetime

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

load_dotenv()


class GraphitiMemory:
    """LangChain-compatible memory interface for Graphiti."""

    def __init__(self, graphiti: Graphiti, group_id: str = 'langchain-agent'):
        self.graphiti = graphiti
        self.group_id = group_id

    async def add_message(self, role: str, content: str):
        """Add message to episodic memory."""
        await self.graphiti.add_episode(
            name=f"conversation_{datetime.now().isoformat()}",
            episode_body=f"{role}: {content}",
            source_description=f"LangChain {role} message",
            group_id=self.group_id,
        )

    async def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """Retrieve relevant context from graph."""
        results = await self.graphiti.search(query, group_ids=[self.group_id])
        
        context_parts = []
        for node in results.nodes[:limit]:
            if isinstance(node, EntityNode):
                context_parts.append(f"Entity: {node.name} - {node.summary}")
            elif isinstance(node, EpisodicNode):
                context_parts.append(f"Memory: {node.content}")
        
        for edge in results.edges[:limit]:
            if isinstance(edge, EntityEdge):
                context_parts.append(
                    f"Relationship: {edge.source_node_uuid} {edge.name} {edge.target_node_uuid}"
                )
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."


async def main():
    """Initialize Graphiti memory backend and demonstrate usage."""
    
    # Load prompts
    with open("examples/integrations/prompts/extraction_prompts.yaml") as f:
        prompts = yaml.safe_load(f)
    
    # Initialize Graphiti
    graphiti = Graphiti(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    await graphiti.build_indices_and_constraints()
    
    memory = GraphitiMemory(graphiti, group_id=os.getenv("GROUP_ID", "langchain-example"))
    
    llm = ChatOpenAI(
        model=prompts["context_retrieval"]["model"],
        temperature=prompts["context_retrieval"]["temperature"],
    )
    
    print("LangChain + Graphiti initialized. Ready for conversations.")
    print("Add your own conversation logic here using memory.add_message() and memory.get_relevant_context()")
    
    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
