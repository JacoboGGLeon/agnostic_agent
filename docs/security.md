# Security Notes

## Secrets policy

- Do not commit real credentials to the repository.
- Store API keys in environment variables or a secret manager.
- Keep tracked config files with placeholders only.

## Local setup

1. Copy `.env.example` values into your local environment.
2. Set `OPENAI_API_KEY` to a valid value in your shell/session.
3. Do not place real secrets in `session/*.yaml` or any tracked file.

## Automated checks

- Use `python -m pytest tests/security -q` to run secret hygiene tests.
- CI should run the same checks and fail on leaked key patterns.
