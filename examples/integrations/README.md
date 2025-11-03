# Integration Examples

Minimal integration examples for Graphiti with popular tools.

## Available Examples

### LangChain Integration
- **File**: `langchain_integration.py`
- **Description**: Graphiti as LangChain memory backend
- **Requirements**: `langchain`, `langchain-openai`, `pyyaml`

### Instructor Integration
- **File**: `instructor_structured_extraction.py`
- **Description**: Type-safe entity extraction with Pydantic
- **Requirements**: `instructor`, `pyyaml`

## Configuration

Prompts are stored in `prompts/extraction_prompts.yaml`. Modify as needed.

### Environment Variables

Set in `.env`:
```bash
# LLM Provider
OPENAI_API_KEY=sk-...

# Graph Database (choose one)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional
MILVUS_HOST=localhost
MILVUS_PORT=19530

GROUP_ID=my-group
TEXT_INPUT="Your text here"  # For instructor example
```

## Usage

```bash
# LangChain
python examples/integrations/langchain_integration.py

# Instructor
TEXT_INPUT="Your text" python examples/integrations/instructor_structured_extraction.py
```

## Customization

- Edit `prompts/extraction_prompts.yaml` for prompts
- Modify connection strings in `.env`
- Add your own conversation/extraction logic in the examples
