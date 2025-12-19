---
name: git-workflow
description: Standard git workflow for feature development
version: 1.0.0
tags:
  - git
  - version-control
  - development
domains:
  - software-development
  - code-management
---

## Overview

This skill covers the standard git workflow for developing features,
fixing bugs, and managing code changes.

## When to Use

- Starting new feature work
- Fixing bugs
- Making code changes that should be tracked
- Preparing code for review

## Steps

1. **Check current status**
   ```bash
   git status
   git branch
   ```

2. **Create feature branch** (if not on one)
   ```bash
   git checkout -b feature/description
   ```

3. **Make changes incrementally**
   - Small, focused commits
   - Clear commit messages explaining "why"

4. **Stage and commit**
   ```bash
   git add <files>
   git commit -m "type: description

   Longer explanation if needed"
   ```

5. **Push and create PR**
   ```bash
   git push -u origin feature/description
   ```

## Watch Out For

- Committing sensitive files (.env, credentials)
- Large binary files
- Committing unfinished work to main branch
- Force pushing to shared branches

## Commit Message Format

```
type: short description

Longer description explaining why this change was made,
not what was changed (the diff shows that).

Refs: #123
```

Types: feat, fix, docs, style, refactor, test, chore
