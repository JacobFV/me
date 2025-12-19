---
name: memory-management
description: Managing agent memory for effective recall
version: 1.0.0
tags:
  - memory
  - self-management
  - agent-specific
domains:
  - agent-operation
type: reflex
---

## Overview

This skill covers how to effectively manage your own memory as an agent.
Your memory directory contains episodes, procedures, theories, and more.
Good memory hygiene improves your effectiveness over time.

## When to Use

- After completing a task (record episode)
- When you discover a reusable pattern (codify procedure)
- When you form a belief about how things work (record theory)
- When something significant happens (mark significant moment)
- When you want to remember something for later (set intention)

## Memory Types

### Episodes (memory/episodes/)
What happened, when, and what you learned.

```markdown
---
id: ep-001
title: Fixed auth timeout bug
outcome: completed
valence: 1
---

## Goal
Fix JWT token expiry issue

## What Happened
Found tokens weren't being refreshed...

## Lessons Learned
- Always check token expiry logic first
```

### Procedures (memory/procedures/)
Reusable patterns you've learned.

```markdown
---
id: proc-001
name: database-migration
times_used: 5
times_succeeded: 4
---

## When to Use
When schema changes are needed

## Steps
1. Create migration file
2. Test locally
3. Backup production
4. Apply migration
```

### Theories (memory/theories/)
Working beliefs about how things work.

```markdown
---
domain: user-behavior
claim: Users abandon forms longer than 3 steps
confidence: 0.7
---

## Reasoning
Based on analytics and user feedback...

## Evidence For
- Conversion rates drop 40% after step 3
```

### Intentions (memory/intentions/)
Things to remember for later.

```markdown
---
reminder: Check deployment logs tomorrow
trigger_description: When working on production
---
```

## Steps

1. **After task completion**: Write episode
2. **Found useful pattern**: Codify procedure
3. **Formed belief**: Record theory
4. **Notable event**: Mark significant
5. **Future action needed**: Set intention

## Watch Out For

- Recording too much (signal vs noise)
- Recording too little (losing valuable learnings)
- Not updating procedures when they fail
- Not revising theories with new evidence
- Forgetting to check intentions

## Using Semantic Memory

The `store_memory` and `recall_memories` tools use vector search:

```
store_memory("JWT tokens expire after 1 hour", tags="auth,tokens")
recall_memories("authentication timeout")
```

This is for quick semantic retrieval. For structured knowledge,
use the markdown files in memory/.
