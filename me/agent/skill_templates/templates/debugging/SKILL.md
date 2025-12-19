---
name: debugging
description: Systematic approach to debugging software issues
version: 1.0.0
tags:
  - debugging
  - problem-solving
  - troubleshooting
domains:
  - software-development
---

## Overview

A systematic approach to finding and fixing bugs. Avoids random
changes and ensures issues are truly fixed, not just hidden.

## When to Use

- Something isn't working as expected
- Tests are failing
- Users report bugs
- Unexpected behavior in production

## Steps

1. **Reproduce the issue**
   - Get exact steps to reproduce
   - Understand expected vs actual behavior
   - Create minimal reproduction if possible

2. **Gather information**
   - Error messages and stack traces
   - Logs around the time of the issue
   - Recent changes that might be related
   - Environment details

3. **Form hypothesis**
   - Based on evidence, what might be wrong?
   - What's the simplest explanation?

4. **Test hypothesis**
   - Add logging/debugging to verify
   - Use debugger to step through
   - Check assumptions

5. **Fix the root cause**
   - Don't just fix symptoms
   - Consider why the bug was possible
   - Prevent similar bugs in future

6. **Verify the fix**
   - Confirm original issue is resolved
   - Check for regressions
   - Add test to prevent recurrence

## Debugging Techniques

### Print/Log Debugging
```python
print(f"DEBUG: variable={variable}, type={type(variable)}")
logger.debug("Reached checkpoint X with state: %s", state)
```

### Binary Search
- Narrow down the problem location by half
- Comment out sections, add early returns
- Git bisect for regression hunting

### Rubber Duck
- Explain the problem out loud
- Often reveals assumptions

### Sleep On It
- Fresh eyes catch what tired eyes miss

## Watch Out For

- Fixing symptoms instead of root cause
- Making multiple changes at once
- Not testing the fix properly
- Assuming you know the cause without evidence
- Ignoring the obvious (typos, wrong file, etc.)

## Common Bug Patterns

- Off-by-one errors
- Null/None reference
- Race conditions
- Type mismatches
- Wrong variable scope
- Cached/stale data
- Environment differences
