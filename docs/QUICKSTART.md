# Quick Start: Contributing to Graphiti

**5-Minute Guide to Making Your First Contribution**

## Pick Your Path

### 🔍 **RAG & Retrieval** (Search Optimization)
- **Easy**: Add alternative fusion method (CombSUM/CombMNZ) to `search_utils.py`
- **Medium**: Implement HyDE query expansion in `search.py`
- **Research**: [Query2doc](https://arxiv.org/abs/2303.07678), [HyDE](https://arxiv.org/abs/2212.10496)

### 🧮 **Embeddings** (Representation Learning)
- **Easy**: Add new embedding provider (Cohere, Jina) to `embedder/`
- **Medium**: Implement Matryoshka dimension reduction
- **Research**: [Matryoshka Embeddings](https://arxiv.org/abs/2205.13147)

### 🔗 **Graph Construction**
- **Easy**: Improve entity deduplication with embedding similarity
- **Medium**: Add structured extraction via Instructor
- **Research**: [Entity Alignment](https://arxiv.org/abs/2206.13163)

### ⚡ **Reranking**
- **Easy**: Add local cross-encoder (ms-marco, bge-reranker)
- **Medium**: Implement learned fusion weights
- **Research**: [BGE Reranker](https://arxiv.org/abs/2309.07597)

### 🌐 **Community Detection**
- **Easy**: Add modularity/conductance metrics
- **Medium**: Implement hierarchical Leiden
- **Research**: [Hierarchical Clustering](https://arxiv.org/abs/1810.08473)

## Setup (3 Steps)

```bash
# 1. Clone and install
git clone https://github.com/getzep/graphiti && cd graphiti
pip install -e ".[dev]"

# 2. Start Neo4j
docker-compose up -d

# 3. Set API keys
export OPENAI_API_KEY=sk-...
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=password
```

## Contribution Workflow

```bash
# 1. Create branch
git checkout -b feature/your-improvement

# 2. Make changes (follow patterns in codebase)
# 3. Test
make format && make lint && make test

# 4. Submit PR with:
#    - Clear description of change
#    - Link to research/issue
#    - Performance comparison (if applicable)
```

## Integration Opportunities

| Tool | Use Case | Location |
|------|----------|----------|
| **LangChain** | Agent memory | `examples/integrations/langchain_integration.py` |
| **Instructor** | Structured extraction | `examples/integrations/instructor_structured_extraction.py` |
| **LlamaIndex** | Custom retriever | *Coming soon* |
| **DSPy** | Prompt optimization | *Coming soon* |

## Key Files

```
graphiti_core/
├── search/
│   ├── search.py              # Main search orchestration
│   ├── search_utils.py        # Search algorithms (BFS, RRF, MMR)
│   └── search_config.py       # Configuration schemas
├── embedder/
│   ├── client.py              # Embedding interface
│   └── openai.py              # OpenAI embeddings
├── llm_client/
│   └── openai_client.py       # LLM interface
└── utils/maintenance/
    ├── node_operations.py     # Entity extraction
    └── community_operations.py # Community detection
```

## Example: Adding a New Embedding Provider

```python
# graphiti_core/embedder/cohere.py
from graphiti_core.embedder.client import EmbedderClient

class CohereEmbedder(EmbedderClient):
    async def create(self, text: str) -> list[float]:
        # Implementation here
        pass
```

## Resources

- **Full Guide**: [docs/contribute.md](./contribute.md) - Deep dive with research citations
- **Examples**: [examples/integrations/](../examples/integrations/) - Working code
- **Discord**: [Join community](https://discord.com/invite/W8Kw6bsgXQ)
- **Paper**: [Zep Architecture](https://arxiv.org/abs/2501.13956)

---

**Ready to contribute?** Start with a "good first issue" or discuss your idea in Discord!
