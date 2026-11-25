# Branch Protection Guidelines

## Recommended Settings for `main` Branch

To safeguard the stability of the production codebase, the following repository settings should be configured in GitHub under **Settings > Branches > Branch protection rules** for `main`:

1. **Require a pull request before merging**:
   - Require at least 1 approval before merging.
   - Dismiss stale pull request approvals when new commits are pushed.

2. **Require status checks to pass before merging**:
   - Require branches to be up to date before merging.
   - Required status checks:
     - `Backend CI / Python 3.11 & Tests`
     - `Frontend CI / TypeScript & Build`
     - `Security / Gitleaks Secret Scan`
     - `Docker / Compose Validation`

3. **Require conversation resolution before merging**:
   - All review comments and discussions must be resolved prior to merge.

4. **Do not allow force pushes**:
   - Ensure `Allow force pushes` is disabled for all users.

5. **Do not allow deletions**:
   - Prevent deletion of the default `main` branch.
