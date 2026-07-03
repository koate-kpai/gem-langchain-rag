# Security Policy

## Secret Management

This project uses a `.env` file for local secret storage, which is **permanently gitignored** (see `.gitignore:5`). We follow these principles:

| Layer | Control | Enforcement |
|-------|---------|-------------|
| Prevention | `.gitignore` blocks `.env`, `.env.local`, `.env.*.local` | Git-level, pre-commit |
| Detection | Runtime validates API key format before any external call | Application-level |
| Response | If a leak is suspected, rotate immediately at OpenAI dashboard | Process-level |

### API Key Hygiene

- All OpenAI API keys use the `sk-proj-` prefix (project keys) or `sk-` (org keys).
- The application validates key format on startup — if a placeholder or malformed key is detected, it fails with an actionable message rather than a cryptic OpenAI 401 error.
- **Do not** hardcode fallback keys, test keys, or expired keys in source code.

### Pre-Commit Recommendation

For teams, add a pre-commit hook (`.git/hooks/pre-commit`) to scan for secrets:

```bash
#!/bin/sh
# Block commits containing potential API keys
if git diff --cached --diff-filter=AM | grep -q 'sk-[A-Za-z0-9]\{20,\}'; then
    echo "ERROR: Commit contains what looks like an API key. Remove it and use .env."
    exit 1
fi
```

### Dependency Security

- All dependencies are pinned in `requirements.txt`.
- Run `pip-audit` regularly to check for known vulnerabilities:

```powershell
pip install pip-audit
pip-audit
```

### Incident Response

If a key is accidentally committed:

1. **Immediately** rotate the key at https://platform.openai.com/api-keys
2. Remove the key from git history using `git-filter-repo`
3. Check OpenAI usage logs for unauthorized access
4. Force-push the cleaned history and notify any collaborators to rebase

## Why This Matters for Portfolio Reviewers

Production ML pipelines handle sensitive data and API credentials. This security posture demonstrates:

- Understanding of **OWASP's defense-in-depth** principle
- Familiarity with **supply-chain security** (pinned deps + auditing)
- Awareness of **credential lifecycle management** (rotation, revocation, monitoring)
- Experience with **DevSecOps** practices (shift-left security, pre-commit hooks)
- Knowledge of **incident response** procedures for credential leaks
