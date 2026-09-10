# Contributing

Contributions that strengthen provenance, retrieval quality, provider portability, or ingestion reliability are welcome.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
```

Pure algorithm tests must not require credentials or live services. Mock provider calls in unit tests and mark any future Neo4j integration suite clearly.

Discuss graph-schema changes before implementation. A schema change should document migration behavior, idempotency, provenance impact, and query compatibility. A new inference rule should include examples of valid and invalid edges.

Keep pull requests focused and update the README when public behavior changes.

## Security reports

Follow [SECURITY.md](SECURITY.md). Do not open a public issue for a vulnerability.
