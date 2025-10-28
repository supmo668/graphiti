.PHONY: install format lint test all check start start-no-neo4j start-mcp start-mcp-with-neo4j start-mcp-with-falkordb

# Define variables
PYTHON = python3
UV = uv
PYTEST = $(UV) run pytest
RUFF = $(UV) run ruff
PYRIGHT = $(UV) run pyright
DOCKER_COMPOSE = docker compose
WITH_NEO4J_PROFILE = with-neo4j
NO_NEO4J_PROFILE = no-neo4j

# Default target
all: format lint test

# Install dependencies
install:
	$(UV) sync --extra dev

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check
	$(PYRIGHT) ./graphiti_core 

# Run tests
test:
	DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 $(PYTEST) -m "not integration"

# Run format, lint, and test
check: format lint test

# Start docker services for the graphiti app with bundled Neo4j
start:
	NEO4J_URI=bolt://neo4j:7687 $(DOCKER_COMPOSE) --profile $(WITH_NEO4J_PROFILE) up --build -d

# Start docker services without the bundled Neo4j container (assumes external Neo4j is running)
start-no-neo4j:
	NEO4J_URI=bolt://neo4j-standalone:7687 $(DOCKER_COMPOSE) up --build -d

# Stop docker services
stop:
	$(DOCKER_COMPOSE) down

# Restart docker services
restart:
	$(DOCKER_COMPOSE) restart

# Start MCP server with bundled Neo4j
start-mcp-with-neo4j:
	NEO4J_URI=bolt://neo4j:7687 DATABASE_TYPE=neo4j $(DOCKER_COMPOSE) --project-directory mcp_server --profile $(WITH_NEO4J_PROFILE) up --build -d

# Start MCP server with bundled FalkorDB
start-mcp-with-falkordb:
	FALKORDB_HOST=falkordb FALKORDB_PORT=6379 DATABASE_TYPE=falkordb INSTALL_FALKORDB=true $(DOCKER_COMPOSE) --project-directory mcp_server --profile with-falkordb up --build -d

# Start MCP server connecting to external Neo4j (assumes Neo4j is running externally)
start-mcp:
	NEO4J_URI=bolt://neo4j-standalone:7687 DATABASE_TYPE=neo4j $(DOCKER_COMPOSE) --project-directory mcp_server up --build -d

# Stop MCP server
stop-mcp:
	$(DOCKER_COMPOSE) --project-directory mcp_server down

# Restart MCP server (uses current .env configuration)
restart-mcp:
	$(DOCKER_COMPOSE) --project-directory mcp_server restart
