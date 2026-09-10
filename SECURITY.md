# Security Policy

Security fixes target the latest release on `main`.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Include reproduction steps, affected versions, and impact. Do not disclose the issue publicly before a fix is available.

Keep provider keys and database credentials in environment variables. Use a least-privileged Neo4j account outside local development, and treat extracted relationship types and model output as untrusted input.
