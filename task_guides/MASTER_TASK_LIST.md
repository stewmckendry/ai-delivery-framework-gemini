# 🧩 MASTER TASK LIST – ProductPod Enhancements

This file outlines the implementation backlog for filling known gaps and improving the ProductPod prototype.

---

## Group A: Frontend Enhancements

### Task 1: Git Diff Preview in Frontend
- Integrate `previewChanges` MCP tool
- Show unified diffs in UI when user proposes changes

### Task 2: Display Tool Results Inline
- Render structured output from Gemini tool calls (e.g., file list, search results) inline in chat
- Format JSON or tabular data clearly

---

## Group B: Auth & Sandbox Flow

### Task 3: GitHub OAuth + Identity
- Implement GitHub OAuth
- Store tokens securely, use per request
- Tie session identity to product pod usage

### Task 4: Sandbox Environment Support
- Add UI flow to create or switch sandbox environments (via `sandboxInit`)
- Let user know whether they’re editing a main or sandbox branch

---

## Group C: Backend Testing + Task Workflow

### Task 5: Tool Test Coverage
- Add tests for all MCP tools in `github_tools.py`
- Use mocking or stub GitHub API clients

### Task 6: Create `task_guides` Templates
- Add templates for task cards and review reports
- Include guidance on input/output, review checklist, etc.

---

## Group D: Deployment + Observability

### Task 7: Deployment Pipeline
- Add GitHub Actions to deploy to Railway
- Include Dockerfile or Railway.toml if needed

### Task 8: Add Metrics Instrumentation
- Log usage events (tool use, commit sizes, errors)
- Use FastMCP context logging or extend it

---

All tasks are tagged in cards using `task_guides/task_<ID>_<desc>.md`. Cards in the same group can be assigned in parallel.
