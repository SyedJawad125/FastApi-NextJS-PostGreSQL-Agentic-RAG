# ============================================================================
# CONTRIBUTING.md
# ============================================================================
# Contributing to Advanced Agentic RAG System

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/agentic-rag.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature`
8. Create a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# Setup environment
make setup
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
make test
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all functions
- Add tests for new features
- Keep functions focused and small

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_agents.py::test_researcher_agent
```

## Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Update documentation
- Add tests for new functionality
- Ensure all tests pass
- Follow existing code style
- Update CHANGELOG.md

## Reporting Issues

- Use issue templates
- Provide clear reproduction steps
- Include error messages and logs
- Specify your environment (OS, Python version)

## Questions?

Open an issue or reach out to the maintainers.
