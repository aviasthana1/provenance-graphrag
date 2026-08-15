"""
Entry point for running the package as a module:
    python -m provenance_graphrag --text-file doc.md --corpus-id abc123
"""

from .cli import main

if __name__ == "__main__":
    main()
