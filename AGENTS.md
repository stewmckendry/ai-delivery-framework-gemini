# 🤖 Codex Agent Guide: ai-delivery-framework-gemini

Welcome to the **Codex Agent workspace** for ProductPod, a cross-functional AI delivery agent.
This file provides structure, expectations, and practices for Codex Agents contributing to this codebase.

---

## 🗂️ Repo Structure

| Folder               | Purpose                                       |
|----------------------|-----------------------------------------------|
| `app/backend/`       | FastAPI backend with Gemini + MCP integration |
| `app/frontend/`      | Simple front end chat UI                      |
| `project/sample/`    | Standalone sample FastAPI project             |
| `tests/`             | Unit tests                                    |
| `task_guides/`       | Task prompts and reports                      |

## 🔁 Naming Conventions

- Task prompts: `task_guides/task_<ID>_<desc>.md`
- Review reports: `task_guides/reports/task_<ID>_<desc>_report.md`
- Code modules follow purpose-specific naming: `github_tools.py`, `main.py`, `script.js`

## ✅ Agent Checklists

Agents must:
- Include input/output schema (use Pydantic)
- Write tests using mock or isolated inputs
- Include example usage or logs
- Summarize logic and tradeoffs in code or PR comment
- Check against `task_guides/review_checklist.md`

## 📦 Requirements

```bash
pip install -r app/backend/requirements.txt
```

## 🤝 Coordination

Agents are expected to:
- Work off `main` or assigned branches
- Commit only assigned or owned files
- Link work to `task_guides/task_<ID>_...` prompt

Thank you for contributing to ProductPod 🚀