# 🧪 Task 002: Inline Tool Output Rendering

## Objective
Display structured outputs from tool calls (e.g. file search, list, fetch) directly in the frontend chat window.

## Context
Tools like `listFiles`, `searchFilesInRepo`, and `fetchFiles` return structured data, but the current chat UI only displays model text.

## Scope
- Parse tool response objects in the frontend
- Display results as formatted JSON blocks or tables
- Preserve role and type in chat history (e.g. `function_response`)

## Inputs
- Tool response payloads from Gemini backend

## Outputs
- Rich output blocks inline in chat window

## Target Files
- `app/frontend/script.js`

## Deliverables
- Parse `function_response` in model replies
- Conditional formatting based on tool (e.g. file list as table)
- Error response fallback (e.g. display error message if tool failed)

## Related
- Gemini output part: `{ "function_response": { "name": ..., "response": { ... } } }`

---

Create a PR and reference this task in the title or body.