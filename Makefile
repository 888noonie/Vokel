.PHONY: all install build start stop dev-frontend clean test help

PORT ?= 8000
HOST ?= 0.0.0.0

help:
	@echo "Voyce Project Commands:"
	@echo "  make install       - Install Python backend dependencies and Frontend node modules"
	@echo "  make build         - Compile TypeScript and bundle frontend assets using Vite"
	@echo "  make start         - Start the FastAPI web application serving the React frontend"
	@echo "  make stop          - Stop any active Voyce web server running on port $(PORT)"
	@echo "  make dev-frontend  - Start the Vite local development server for frontend HMR"
	@echo "  make test          - Run full Python test suite with pytest"
	@echo "  make clean         - Clear built assets, node_modules, and virtualenv/python caches"

install:
	@echo "Installing Python dependencies with uv..."
	uv pip install -e ".[dev,audio]"
	@echo "Installing Frontend node modules..."
	cd frontend && npm install

build:
	@echo "Building frontend static assets..."
	@test -d frontend/node_modules || (echo "Installing frontend dependencies (npm install)..." && cd frontend && npm install)
	cd frontend && npm run build

start: build
	@echo "Starting Voyce Web Server on http://$(HOST):$(PORT)..."
	.venv/bin/python -m voyce.cli --web --host $(HOST) --port $(PORT)

stop:
	@echo "Stopping any process running on port $(PORT)..."
	@PID=$$(lsof -t -i:$(PORT) -sTCP:LISTEN); \
	if [ -n "$$PID" ]; then \
		echo "Found process $$PID on port $(PORT). Terminating..."; \
		kill $$PID || kill -9 $$PID; \
		echo "Process stopped."; \
	else \
		echo "No active process found listening on port $(PORT)."; \
	fi

dev-frontend:
	@echo "Starting Vite local development server..."
	cd frontend && npm run dev

test:
	@echo "Running tests..."
	PYTHONPATH=. .venv/bin/pytest -v

clean:
	@echo "Cleaning caches and builds..."
	rm -rf frontend/dist
	rm -rf frontend/node_modules
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	@echo "Clean completed."
