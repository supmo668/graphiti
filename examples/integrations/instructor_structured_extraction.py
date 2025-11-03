"""
Instructor Integration - Structured Entity Extraction

Demonstrates type-safe extraction using Instructor with Graphiti.
See README.md for setup instructions.
"""

import asyncio
import os
from typing import List

import instructor
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from graphiti_core import Graphiti


class Entity(BaseModel):
    """Entity extracted from text."""
    name: str = Field(..., description="Entity name or identifier")
    entity_type: str = Field(..., description="Entity type (Person, Organization, Location, etc.)")
    summary: str = Field(..., description="Brief description")
    attributes: dict[str, str] = Field(default_factory=dict, description="Additional attributes")


class Relation(BaseModel):
    """Relationship between entities."""
    source_entity: str = Field(..., description="Source entity name")
    relation_type: str = Field(..., description="Relationship type")
    target_entity: str = Field(..., description="Target entity name")
    description: str = Field(..., description="Relationship description")


class ExtractedKnowledge(BaseModel):
    """Complete extraction result."""
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    relations: List[Relation] = Field(default_factory=list, description="Extracted relations")


class StructuredExtractor:
    """Type-safe extractor using Instructor."""
    
    def __init__(self, api_key: str | None = None, config: dict | None = None):
        self.client = instructor.from_openai(
            AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        )
        self.config = config or {}
    
    async def extract_knowledge(self, text: str) -> ExtractedKnowledge:
        """Extract entities and relations with schema validation."""
        response = await self.client.chat.completions.create(
            model=self.config.get("model", "gpt-4o-mini"),
            response_model=ExtractedKnowledge,
            messages=[
                {"role": "system", "content": self.config.get("system_prompt", "")},
                {"role": "user", "content": text}
            ],
            temperature=self.config.get("temperature", 0.3),
        )
        return response


async def main():
    """Initialize structured extractor and demonstrate usage."""
    
    # Load prompts
    with open("examples/integrations/prompts/extraction_prompts.yaml") as f:
        prompts = yaml.safe_load(f)
    
    extractor = StructuredExtractor(config=prompts["extraction"])
    
    # Initialize Graphiti
    graphiti = Graphiti(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    await graphiti.build_indices_and_constraints()
    
    print("Instructor + Graphiti initialized. Ready for structured extraction.")
    print("Provide your own text via environment variable TEXT_INPUT or modify this file.")
    
    # Get text from environment or use minimal placeholder
    text_input = os.getenv("TEXT_INPUT")
    if text_input:
        knowledge = await extractor.extract_knowledge(text_input)
        print(f"Extracted {len(knowledge.entities)} entities and {len(knowledge.relations)} relations")
    
    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
