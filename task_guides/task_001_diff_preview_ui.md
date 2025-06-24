# 🧪 Task 001: Git Diff Preview in Frontend

## Objective
Enable users to preview GitHub file diffs directly from the frontend before committing changes.

## Context
The backend supports a `previewChanges` MCP tool that generates a unified diff string. However, this is not yet connected to the frontend chat interface.

## Scope
- Add UI control (e.g. "Preview Changes" button)
- Call `/mcp/previewChanges` tool with current file edits
- Display returned diff in readable format (e.g. monospace block or syntax-highlighted)
- Update chat history with diff block

## Inputs
- Existing edited file content (path + new content)

## Outputs
- Unified diff preview shown in chat window

## Target Files
- `app/frontend/script.js`
- Optionally: `index.html` (if new UI elements added)

## Deliverables
- Clean UI integration for diff
- Graceful error handling
- Inline message formatting for diff result

## Related
- See `github_tools.py` tool: `preview_changes_mcp_tool`

---

Create a PR and reference this task in the title or body.