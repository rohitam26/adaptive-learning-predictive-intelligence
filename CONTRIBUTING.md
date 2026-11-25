# Contributing to Adaptive Learning Platform

Thank you for your interest in contributing to the Adaptive Learning and Predictive Intelligence Platform.

## Code of Conduct
All contributors and participants are expected to maintain a welcoming, respectful, and constructive environment.

## Development Workflow

1. **Fork and Branch**:
   - Create a feature branch from `main`:
     ```bash
     git checkout -b feat/your-feature-name
     ```

2. **Environment Setup**:
   - Python: Ensure Python 3.11 is installed. Synchronize virtual environment with `uv sync`.
   - Node: Ensure Node.js 20+ is installed. Run `npm install` inside `frontend/`.

3. **Code Style & Quality**:
   - Backend: Run `ruff check backend ml` and ensure zero errors.
   - Frontend: Run `npm run build` in `frontend/` to confirm TypeScript and Vite compilation.
   - Tests: Run `pytest` and verify all tests pass.

4. **Commit Guidelines**:
   - Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).

5. **Submitting Pull Requests**:
   - Ensure the pull request template is completed with a clear description and testing evidence.
   - Verify that automated GitHub Actions CI checks succeed.
