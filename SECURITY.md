# Security Policy & Vulnerability Reporting

## 1. Supported Versions

Security updates and patches are applied to the active `main` branch.

| Version / Branch | Supported          |
| ---------------- | ------------------ |
| `main`           | :white_check_mark: |
| Releases         | :white_check_mark: |
| Historical tags  | :x:                |

---

## 2. Reporting a Vulnerability

The project team takes security vulnerabilities seriously. If you discover a security issue or suspected vulnerability, **please do not open a public GitHub issue**.

### Preferred Reporting Method:
1. **GitHub Private Vulnerability Reporting**:
   - Navigate to the **Security** tab of this repository on GitHub.
   - Click **Report a vulnerability** to open an advisory draft.
2. **Direct Maintainer Contact**:
   - Email: **rohitam256@gmail.com**
   - Subject: `[SECURITY] Adaptive Learning Platform - Vulnerability Report`

### Information to Include in Your Report:
- Detailed description of the vulnerability.
- Steps to reproduce the issue (proof-of-concept script, request payload, or test case).
- Potential impact on learner privacy, database integrity, or authentication bypass.
- Any suggested fixes or mitigations.

### Response Timeline:
- **Acknowledgment**: Within 48 hours of receipt.
- **Assessment & Status Update**: Within 5 business days.
- **Remediation & Advisory Release**: Coordinated with the reporter prior to public disclosure.

---

## 3. Secret & Credential Rotation Protocol

In the event that an API key, database credential, or JWT signing secret is accidentally exposed:

1. **Immediate Revocation**:
   - Invalidate the compromised token or credential at the provider / identity service immediately.
2. **Re-key Application Secrets**:
   - Generate a cryptographically random secret (e.g. `openssl rand -hex 32`).
   - Update deployment environment variables (`SECRET_KEY`, `POSTGRES_PASSWORD`).
   - Restart backend services to invalidate all existing sessions.
3. **Repository Cleanliness**:
   - If a commit contains a secret, use `git filter-repo` or BFG Repo-Cleaner to expunge it from Git history.
   - Force secret rotation even if the commit history is rewritten.
4. **Audit Logs**:
   - Review database access logs and authentication audits for anomalous activity during the exposure window.

---

## 4. Development Security Best Practices

- **Zero Hardcoded Secrets**: Secrets are loaded strictly via environment variables parsed by Pydantic Settings (`backend/app/core/config.py`).
- **Pre-commit Scanning**: Gitleaks and pre-commit hooks inspect staged changes locally before commits are accepted.
- **Dependency Auditing**: Automated GitHub Actions run `pip-audit` and `npm audit` to identify vulnerable dependencies.
