# Integration Examples

This directory contains examples of integrating Graphiti with popular open-source tools and frameworks.

## Available Examples

### LangChain Integration
- **File**: `langchain_integration.py`
- **Description**: Use Graphiti as a memory backend for LangChain agents
- **Tools Used**: LangChain, langchain-mcp-adapter (optional)
- **Use Case**: Persistent agent memory with temporal context

### Instructor Integration
- **File**: `instructor_structured_extraction.py`
- **Description**: Type-safe entity and relation extraction using Instructor
- **Tools Used**: Instructor, Pydantic
- **Use Case**: Structured, validated knowledge extraction

### DSPy Integration
- **File**: `dspy_prompt_optimization.py` (Coming Soon)
- **Description**: Optimize extraction prompts using DSPy
- **Tools Used**: DSPy
- **Use Case**: Automated prompt engineering for better extraction quality

### LlamaIndex Integration
- **File**: `llamaindex_retriever.py` (Coming Soon)
- **Description**: Custom LlamaIndex retriever backed by Graphiti
- **Tools Used**: LlamaIndex
- **Use Case**: Graph-based RAG within LlamaIndex pipelines

## Setup

Install optional dependencies for integrations:

```bash
# LangChain
pip install langchain langchain-openai

# Instructor
pip install instructor

# DSPy
pip install dspy-ai

# LlamaIndex
pip install llama-index
```

## Running Examples

Each example is self-contained and can be run independently:

```bash
# Set up environment variables (see .env.example)
export OPENAI_API_KEY=sk-...
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# Run example
python examples/integrations/langchain_integration.py
```

## Contributing

Have an integration example to share? Follow these steps:

1. Create a new file in this directory
2. Include docstrings explaining the use case
3. Add setup instructions and dependencies
4. Ensure it runs without errors
5. Submit a PR with your example

See [docs/contribute.md](../../docs/contribute.md) for detailed contribution guidelines.
