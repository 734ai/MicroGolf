# MicroGolf: Ultra-Compact ARC-AGI Solution Framework
# NeurIPS 2025 Google Code Golf Championship

.PHONY: help setup test lint format check-bytes ci-report clean

# Configuration
PROJECT_NAME := microgolf
PYTHON := python3
PIP := pip3
MAX_BYTES := 2500
SUBMISSION_DIR := submission
DATA_DIR := data/arc
EXPERIMENTS_DIR := experiments

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)MicroGolf - Ultra-Compact ARC Solution Framework$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install dependencies and setup environment
	@echo "$(BLUE)Setting up MicroGolf environment...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	mkdir -p $(DATA_DIR) $(EXPERIMENTS_DIR) $(SUBMISSION_DIR)
	mkdir -p logs model/checkpoints notebooks/outputs
	@echo "$(GREEN)Setup complete!$(NC)"

setup-dev: setup ## Setup development environment with additional tools
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PIP) install pytest pytest-cov black flake8 mypy pre-commit wandb astor
	pre-commit install
	@echo "$(GREEN)Development setup complete!$(NC)"

test: ## Run unit tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term
	@echo "$(GREEN)Tests completed!$(NC)"

test-fast: ## Run fast tests only
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -k "not slow"

lint: ## Run code linting
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 $(PROJECT_NAME)/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy $(PROJECT_NAME)/ --ignore-missing-imports
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(PROJECT_NAME)/ scripts/ tests/ --line-length=100
	@echo "$(GREEN)Formatting complete!$(NC)"

check-bytes: ## Check byte count compliance for all solutions
	@echo "$(BLUE)Checking byte count compliance...$(NC)"
	@total_violations=0; \
	for file in $(SUBMISSION_DIR)/*.py; do \
		if [ -f "$$file" ]; then \
			bytes=$$(wc -c < "$$file"); \
			if [ $$bytes -gt $(MAX_BYTES) ]; then \
				echo "$(RED)❌ $$file: $$bytes bytes (exceeds $(MAX_BYTES))$(NC)"; \
				total_violations=$$((total_violations + 1)); \
			else \
				echo "$(GREEN)✅ $$file: $$bytes bytes$(NC)"; \
			fi; \
		fi; \
	done; \
	if [ $$total_violations -eq 0 ]; then \
		echo "$(GREEN)All files comply with $(MAX_BYTES) byte limit!$(NC)"; \
	else \
		echo "$(RED)$$total_violations files exceed byte limit$(NC)"; \
		exit 1; \
	fi

optimize: ## Run code optimization on all solutions
	@echo "$(BLUE)Optimizing solutions...$(NC)"
	$(PYTHON) scripts/prune_characters.py $(SUBMISSION_DIR) --output_dir $(SUBMISSION_DIR)_optimized --max_bytes $(MAX_BYTES) --report optimization_report.json
	@echo "$(GREEN)Optimization complete! Check optimization_report.json$(NC)"

train-meta: ## Train the meta-composer model
	@echo "$(BLUE)Training meta-composer...$(NC)"
	mkdir -p $(EXPERIMENTS_DIR)/meta
	$(PYTHON) scripts/train_meta_controller.py \
		--data_dir $(DATA_DIR) \
		--output_dir $(EXPERIMENTS_DIR)/meta \
		--epochs 50 \
		--batch_size 16 \
		--learning_rate 1e-3 \
		--use_wandb
	@echo "$(GREEN)Training complete!$(NC)"

evaluate: ## Evaluate current solutions on ARC tasks
	@echo "$(BLUE)Evaluating solutions...$(NC)"
	$(PYTHON) scripts/eval_submission.py \
		--data_dir $(DATA_DIR) \
		--model_path $(EXPERIMENTS_DIR)/meta/best_model.pt \
		--output_file evaluation_results.json \
		--sample_size 100
	@echo "$(GREEN)Evaluation complete! Check evaluation_results.json$(NC)"

generate-submission: ## Generate final submission files
	@echo "$(BLUE)Generating submission...$(NC)"
	mkdir -p $(SUBMISSION_DIR)
	$(PYTHON) scripts/generate_submission.py \
		--data_dir $(DATA_DIR) \
		--model_path $(EXPERIMENTS_DIR)/meta/best_model.pt \
		--output_dir $(SUBMISSION_DIR) \
		--max_bytes $(MAX_BYTES)
	@echo "$(GREEN)Submission generated in $(SUBMISSION_DIR)/$(NC)"

package-submission: generate-submission optimize check-bytes ## Package final submission
	@echo "$(BLUE)Packaging final submission...$(NC)"
	cd $(SUBMISSION_DIR) && zip -r ../submission.zip *.py
	@echo "$(GREEN)Submission packaged as submission.zip$(NC)"
	@echo "$(YELLOW)Files in submission:$(NC)"
	@unzip -l submission.zip

ci-report: ## Generate comprehensive CI report
	@echo "$(BLUE)Generating CI report...$(NC)"
	@echo "# MicroGolf CI Report" > ci_report.md
	@echo "Generated: $$(date)" >> ci_report.md
	@echo "" >> ci_report.md
	
	@echo "## Code Quality" >> ci_report.md
	@echo '```' >> ci_report.md
	@$(MAKE) lint 2>&1 | tee -a ci_report.md || true
	@echo '```' >> ci_report.md
	@echo "" >> ci_report.md
	
	@echo "## Test Results" >> ci_report.md
	@echo '```' >> ci_report.md
	@$(MAKE) test-fast 2>&1 | tee -a ci_report.md || true
	@echo '```' >> ci_report.md
	@echo "" >> ci_report.md
	
	@echo "## Byte Count Compliance" >> ci_report.md
	@echo '```' >> ci_report.md
	@$(MAKE) check-bytes 2>&1 | tee -a ci_report.md || true
	@echo '```' >> ci_report.md
	@echo "" >> ci_report.md
	
	@echo "$(GREEN)CI report generated: ci_report.md$(NC)"

benchmark: ## Run comprehensive benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) scripts/benchmark.py --output benchmark_results.json
	@echo "$(GREEN)Benchmarks complete! Check benchmark_results.json$(NC)"

profile: ## Profile meta-composer performance
	@echo "$(BLUE)Profiling meta-composer...$(NC)"
	$(PYTHON) -m cProfile -o profile_stats.prof scripts/train_meta_controller.py --epochs 1
	$(PYTHON) -c "import pstats; pstats.Stats('profile_stats.prof').sort_stats('cumulative').print_stats(20)"

demo: ## Run demonstration notebook
	@echo "$(BLUE)Running demo...$(NC)"
	jupyter nbconvert --execute --to html notebooks/01_microgolf_demo.ipynb --output ../demo_output.html
	@echo "$(GREEN)Demo complete! Check demo_output.html$(NC)"

clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf build dist *.egg-info
	rm -rf $(EXPERIMENTS_DIR)/*/checkpoints/*.pt
	rm -f *.prof *.log optimization_report.json evaluation_results.json
	rm -f submission.zip ci_report.md demo_output.html
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-all: clean ## Clean everything including data and models
	@echo "$(YELLOW)Warning: This will delete all experiments and data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(DATA_DIR) $(EXPERIMENTS_DIR) $(SUBMISSION_DIR); \
		echo "$(GREEN)Full cleanup complete!$(NC)"; \
	else \
		echo "$(BLUE)Cleanup cancelled$(NC)"; \
	fi

install: ## Install the package
	$(PIP) install -e .

docker-build: ## Build Docker container
	docker build -t microgolf:latest .

docker-run: ## Run in Docker container
	docker run -it --rm -v $(PWD):/workspace microgolf:latest bash

status: ## Show project status
	@echo "$(BLUE)MicroGolf Project Status$(NC)"
	@echo ""
	@echo "$(YELLOW)Repository:$(NC)"
	@git status --short 2>/dev/null || echo "Not a git repository"
	@echo ""
	@echo "$(YELLOW)Files:$(NC)"
	@echo "  Python files: $$(find . -name "*.py" | wc -l)"
	@echo "  Test files: $$(find tests -name "*.py" 2>/dev/null | wc -l || echo 0)"
	@echo "  Notebooks: $$(find notebooks -name "*.ipynb" 2>/dev/null | wc -l || echo 0)"
	@echo ""
	@echo "$(YELLOW)Submission:$(NC)"
	@if [ -d "$(SUBMISSION_DIR)" ]; then \
		echo "  Solution files: $$(ls $(SUBMISSION_DIR)/*.py 2>/dev/null | wc -l || echo 0)"; \
		total_bytes=$$(find $(SUBMISSION_DIR) -name "*.py" -exec wc -c {} + 2>/dev/null | tail -1 | cut -d' ' -f1 || echo 0); \
		echo "  Total bytes: $$total_bytes"; \
	else \
		echo "  No submission directory found"; \
	fi
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@if [ -d "$(EXPERIMENTS_DIR)" ]; then \
		echo "  Checkpoints: $$(find $(EXPERIMENTS_DIR) -name "*.pt" 2>/dev/null | wc -l || echo 0)"; \
	else \
		echo "  No experiments directory found"; \
	fi

watch-train: ## Watch training progress (requires tmux)
	@if command -v tmux >/dev/null 2>&1; then \
		tmux new-session -d -s microgolf-train "$(MAKE) train-meta"; \
		echo "$(GREEN)Training started in tmux session 'microgolf-train'$(NC)"; \
		echo "$(BLUE)Use 'tmux attach -t microgolf-train' to view progress$(NC)"; \
	else \
		echo "$(RED)tmux not found. Please install tmux or run 'make train-meta' directly$(NC)"; \
	fi

# Advanced targets for competition preparation
competition-prep: clean setup-dev test lint check-bytes ## Prepare for competition submission
	@echo "$(GREEN)Competition preparation complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Run: make train-meta"
	@echo "  2. Run: make evaluate"
	@echo "  3. Run: make package-submission"

final-submission: train-meta evaluate package-submission ## Create final submission (full pipeline)
	@echo "$(GREEN)Final submission ready!$(NC)"
	@echo "$(BLUE)Submission details:$(NC)"
	@ls -la submission.zip
	@echo ""
	@echo "$(YELLOW)Upload submission.zip to Kaggle$(NC)"

# Help text
.DEFAULT_GOAL := help
