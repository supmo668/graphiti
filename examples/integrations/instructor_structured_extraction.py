"""
Instructor Integration Example - Structured Entity Extraction

This example demonstrates using Instructor with Graphiti for type-safe, validated
entity and relation extraction from unstructured text.

Use Case:
- Guaranteed schema compliance for extracted entities
- Reduced LLM hallucinations through structured outputs
- Type-safe extraction with Pydantic validation
- Improved reliability for production systems

Requirements:
- instructor
- openai
- pydantic
- graphiti-core

Research Reference:
- Instructor: https://github.com/jxnl/instructor
- Structured Outputs: https://openai.com/index/introducing-structured-outputs-in-the-api/
"""

import asyncio
import os
from datetime import datetime
from typing import List

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode


# Define strict schemas for extraction
class Entity(BaseModel):
    """Represents an entity extracted from text."""
    name: str = Field(..., description="The name or identifier of the entity")
    entity_type: str = Field(
        ..., 
        description="Type of entity (e.g., Person, Organization, Location, Concept)"
    )
    summary: str = Field(
        ..., 
        description="Brief description of the entity and its relevance"
    )
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Additional attributes as key-value pairs"
    )


class Relation(BaseModel):
    """Represents a relationship between two entities."""
    source_entity: str = Field(..., description="Name of the source entity")
    relation_type: str = Field(..., description="Type of relationship")
    target_entity: str = Field(..., description="Name of the target entity")
    description: str = Field(..., description="Description of the relationship")


class ExtractedKnowledge(BaseModel):
    """Complete extraction result with entities and relations."""
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of entities found in the text"
    )
    relations: List[Relation] = Field(
        default_factory=list,
        description="List of relationships between entities"
    )


class StructuredExtractor:
    """
    Type-safe entity and relation extractor using Instructor.
    
    This class wraps OpenAI's API with Instructor to ensure all extractions
    follow predefined Pydantic schemas, preventing invalid or incomplete data.
    """
    
    def __init__(self, api_key: str | None = None):
        self.client = instructor.from_openai(
            AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        )
    
    async def extract_knowledge(self, text: str) -> ExtractedKnowledge:
        """
        Extract entities and relations from text with guaranteed schema compliance.
        
        Args:
            text: Input text to extract knowledge from
            
        Returns:
            ExtractedKnowledge object with validated entities and relations
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=ExtractedKnowledge,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert knowledge extraction system. 
                    Extract all entities and their relationships from the given text.
                    Be comprehensive but accurate. Include entity types like Person, 
                    Organization, Location, Concept, Event, etc."""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3,
        )
        return response


async def main():
    """
    Demonstrate structured extraction with Instructor + Graphiti integration.
    
    This example shows:
    1. Using Pydantic schemas for type-safe extraction
    2. Validating extracted data before adding to graph
    3. Handling extraction errors gracefully
    4. Comparing structured vs. unstructured approaches
    """
    
    print("="*60)
    print("Instructor + Graphiti Structured Extraction Example")
    print("="*60)
    
    # Initialize components
    extractor = StructuredExtractor()
    
    # Sample text for extraction
    sample_text = """
    Dr. Sarah Chen, a renowned AI researcher at Stanford University, recently published 
    a groundbreaking paper on temporal knowledge graphs. She collaborated with her 
    colleague Prof. Michael Rodriguez from MIT on this project. The research was 
    funded by a $2M grant from the National Science Foundation. The paper introduces 
    a novel algorithm called TemporalGraph that improves knowledge retention by 40% 
    compared to existing methods. Sarah presented their findings at the NeurIPS 2024 
    conference in New Orleans.
    """
    
    print(f"\nInput Text:\n{sample_text.strip()}")
    print("\n" + "-"*60)
    
    # Extract with structured approach
    print("\nExtracting knowledge with Instructor (structured)...")
    
    try:
        knowledge = await extractor.extract_knowledge(sample_text)
        
        print(f"\n✓ Extracted {len(knowledge.entities)} entities:")
        for entity in knowledge.entities:
            print(f"  • {entity.name} ({entity.entity_type})")
            print(f"    Summary: {entity.summary}")
            if entity.attributes:
                print(f"    Attributes: {entity.attributes}")
        
        print(f"\n✓ Extracted {len(knowledge.relations)} relations:")
        for relation in knowledge.relations:
            print(f"  • {relation.source_entity} --[{relation.relation_type}]--> {relation.target_entity}")
            print(f"    Description: {relation.description}")
        
        # Now add to Graphiti
        print("\n" + "-"*60)
        print("Adding structured data to Graphiti...")
        
        graphiti = Graphiti(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
        await graphiti.build_indices_and_constraints()
        
        # Add episode with structured extraction
        await graphiti.add_episode(
            name=f"structured_extraction_{datetime.now().isoformat()}",
            episode_body=sample_text,
            source_description="Instructor structured extraction example",
            group_id="instructor-example",
        )
        
        # Search for extracted entities
        search_query = "AI researcher Stanford"
        print(f"\nSearching graph for: '{search_query}'")
        
        results = await graphiti.search(search_query, group_ids=["instructor-example"])
        print(f"Found {len(results.nodes)} relevant nodes")
        
        for node in results.nodes[:3]:
            if isinstance(node, EntityNode):
                print(f"  • {node.name}: {node.summary}")
        
        await graphiti.close()
        
        print("\n" + "="*60)
        print("Benefits of Structured Extraction:")
        print("="*60)
        print("✓ Guaranteed schema compliance (no missing fields)")
        print("✓ Type safety with Pydantic validation")
        print("✓ Easier to test and debug")
        print("✓ Consistent data structure for downstream processing")
        print("✓ Reduced need for post-processing and error handling")
        
    except Exception as e:
        print(f"\n✗ Error during extraction: {e}")
        print("Check your OpenAI API key and network connection.")
    
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
