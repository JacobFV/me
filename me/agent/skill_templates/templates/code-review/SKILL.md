---
name: code-review
description: Systematic code review checklist
version: 1.0.0
tags:
  - code-quality
  - review
  - best-practices
domains:
  - software-development
---

## Overview

A systematic approach to reviewing code changes, whether self-review
before committing or reviewing others' pull requests.

## When to Use

- Before committing code
- Reviewing pull requests
- Auditing existing code
- After making significant changes

## Review Checklist

### 1. Correctness
- [ ] Does the code do what it's supposed to do?
- [ ] Are edge cases handled?
- [ ] Are there any obvious bugs?
- [ ] Do tests pass?

### 2. Security
- [ ] No hardcoded secrets or credentials
- [ ] Input validation present
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] Authentication/authorization checks

### 3. Performance
- [ ] No N+1 query problems
- [ ] Appropriate caching
- [ ] No unnecessary loops or allocations
- [ ] Reasonable algorithmic complexity

### 4. Maintainability
- [ ] Clear naming (variables, functions, classes)
- [ ] Single responsibility (each function does one thing)
- [ ] No excessive complexity
- [ ] Reasonable file/function length

### 5. Testing
- [ ] Tests cover happy path
- [ ] Tests cover edge cases
- [ ] Tests are readable and maintainable
- [ ] No flaky tests

### 6. Documentation
- [ ] Complex logic is commented
- [ ] Public APIs are documented
- [ ] README updated if needed

## Steps

1. **Understand the context** - What problem is being solved?
2. **Read the diff** - Understand what changed
3. **Run through checklist** - Check each category
4. **Test locally** - If significant changes
5. **Provide feedback** - Be constructive and specific

## Watch Out For

- Being too nitpicky about style (use linters)
- Missing the forest for the trees
- Not understanding the broader context
- Rubber-stamping without actually reviewing
