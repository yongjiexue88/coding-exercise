---
description: Safe, deterministic commit with Conventional Commits messages. Blocks secrets, build artifacts, and binaries.
---

# /commit — Safe Deterministic Commit

> **Goal** — Stage only safe, intended files · Block secrets + build artifacts + binaries · Create a Conventional Commits message with the correct scope · Provide copy-paste commands.

---

## Hard-Stop Conditions (non-negotiable)

Before staging or committing, check for the following. If any condition is met, **STOP** and clearly report it.

| # | Condition | Action if **staged** | Action if **untracked / worktree only** |
|---|-----------|---------------------|-----------------------------------------|
| 1 | **Secrets / credentials** — `.env`, `.env.*` (except `.env.example`), `service-account.json`, `*service_account*.json`, private keys, tokens, Firebase Admin keys, Stripe secrets, OAuth secrets | **STOP** — unstage immediately | **WARN** — keep unstaged |
| 2 | **Build artifacts / binaries** — `dist/`, `build/`, `coverage/`, `.next/`, `electron/out/`, `node_modules/`, `__pycache__/`, `venv/`, `.venv/`, `chroma_db/`, `*.dmg`, `*.exe`, `*.app`, `*.deb`, `*.rpm`, `*.egg`, `*.pyc` | **STOP** — unstage immediately (unless user explicitly asked) | **WARN** — keep unstaged |
| 3 | **Nothing staged** — after staging + cleanup, no files remain staged | **STOP** — do not create an empty commit | — |

---

## Step 0 — Preflight: Show Status + Change Summary

// turbo
```bash
git status -sb
```

// turbo
```bash
git diff --stat
```

// turbo
```bash
git diff --name-only
```

// turbo
```bash
git diff --name-only --cached
```

**Evaluate hard-stops:**
- If anything forbidden is already staged → run the **Emergency Unstage** block (Step E) before continuing.
- Print a summary: total changed files, any warnings for untracked forbidden files.

---

## Step 1 — Deterministic Staging

Stage only the known safe roots that exist in this repo. Only include directories/files that actually exist.

```bash
git add -A -- frontend backend .github docker-compose.yml README.md .gitignore
```

> **Note:** If `electron/`, `.github/`, or other roots don't exist, omit them from the command — `git add` will error on missing paths.

---

## Step 2 — Immediately Unstage Forbidden / Generated Paths

// turbo
```bash
git reset -- electron/out dist build coverage node_modules __pycache__ venv .venv chroma_db 2>/dev/null; \
git reset -- '**/*.dmg' '**/*.exe' '**/*.app' '**/*.deb' '**/*.rpm' '**/*.egg' '**/*.pyc' 2>/dev/null; \
git reset -- '.env' '.env.*' 'backend/.env' 'service-account.json' '*service_account*.json' 2>/dev/null; \
echo "--- Unstage sweep complete ---"
```

---

## Step 3 — Verify Staged Set (must be clean)

// turbo
```bash
git diff --name-only --cached
```

**Evaluate hard-stops:**
- If **any** forbidden path appears → **STOP** and run Emergency Unstage (Step E) again.
- If the list is **empty** → **STOP** — nothing to commit.
- Otherwise print the clean staged file list and continue.

---

## Step 4 — Generate Commit Message

Use the **Conventional Commits** format:

```
type(scope): short imperative summary
```

### Types
`feat` · `fix` · `refactor` · `perf` · `test` · `docs` · `chore` · `build` · `ci`

### Scopes (pick the most specific one)
`frontend` · `backend` · `electron` · `auth` · `stripe` · `firebase` · `ci` · `infra` · `deps` · `docker` · `docs`

> If changes span multiple scopes equally, use the broadest applicable scope or omit scope: `chore: description`.

### Rules for the summary line
- Imperative mood ("add", not "added" or "adds")
- Lowercase start, no trailing period
- ≤ 72 characters

**Generate 3–6 suggested messages** based on the staged file list, for example:

```
fix(backend): handle null session state in socket auth
feat(frontend): add operator standby status indicator
refactor(backend): simplify registration logging flow
docs(docs): document release tagging and build steps
chore(deps): bump socket.io dependencies
test(backend): add regression coverage for login dial flow
```

Present them as a numbered list so the user can pick one or request changes.

---

## Step 5 — Confirm Staged Is Non-Empty

// turbo
```bash
git diff --name-only --cached | head -20
```

If empty → **STOP**.

---

## Step 6 — Commit

Use the selected (or best) message. Include a body with **what changed + why**.

```bash
git commit -m "type(scope): summary" -m "What changed and why in 1-2 sentences."
```

---

## Step 7 — Post-Commit Verification

// turbo
```bash
git status -sb
```

// turbo
```bash
git show --stat --oneline -1
```

---

## Step E — Emergency Unstage (run if bad stuff got staged)

// turbo
```bash
git reset -- .env '.env.*' 'backend/.env' service-account.json '*service_account*.json' 2>/dev/null
git reset -- electron/out dist build coverage node_modules __pycache__ venv .venv chroma_db 2>/dev/null
git reset -- '**/*.dmg' '**/*.exe' '**/*.app' '**/*.deb' '**/*.rpm' '**/*.egg' '**/*.pyc' 2>/dev/null
git diff --name-only --cached
```

---

## Output Checklist

Every `/commit` run **must** print:

- [ ] `git status -sb` summary
- [ ] Staged file list (`git diff --name-only --cached`)
- [ ] Any **WARN** / **STOP** findings (forbidden files)
- [ ] Copy-paste staging commands used
- [ ] 3–6 suggested Conventional Commit messages
- [ ] Final commit command (ready to paste or executed)
- [ ] Post-commit `git show --stat -1`
