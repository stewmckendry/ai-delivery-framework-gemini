# ai-delivery-framework-gemini

This repository is a ProductPod prototype built with a FastAPI backend, Google Gemini LLM integration, and a minimal HTML+JS frontend. It includes MCP tools to securely access and modify GitHub repositories.

## 🧠 What is ProductPod?
ProductPod is a cross-functional AI agent designed to:
- Translate goals into actionable task cards
- Search, fetch, and update GitHub files
- Guide Codex Agents with clear scope and instructions
- Run on Gemini with support for tool calling (via FastMCP)

## 📁 Structure

```
app/
├── backend/         # FastAPI app, Gemini proxy, GitHub tools
│   └── main.py      # Main app logic and Gemini function handling
│   └── github_tools.py  # MCP tools for file list, fetch, search, commit
├── frontend/        # Lightweight chat interface
│   └── index.html
│   └── script.js
project/sample/      # Sample project (for Gemini tool discovery demo)
task_guides/         # Prompts, review checklists (planned)
```

## 🚀 Running Locally

1. **Install dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/backend/requirements.txt
```

2. **Create a `.env` file**
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-1.5-flash-latest
ALLOWED_ORIGINS=*
```

3. **Start the backend**
```bash
uvicorn app.backend.main:app --reload
```

4. **Open the frontend**
```bash
open app/frontend/index.html
```

## 🔌 MCP Tools
The backend exposes tools via `/mcp` that can be used by Gemini:
- `fetchFiles`
- `listFiles`
- `searchFilesInRepo`
- `gitCommit`
- `previewChanges`
- `sandboxInit`

These are automatically exposed to Gemini for function calling.

## 🧪 Testing
Run backend unit tests:
```bash
pytest
```

## 🧭 For Codex Agents
See [AGENTS.md](AGENTS.md) for structure, contribution guides, and naming conventions.

## 🌐 Deployment
Planned support for Railway and Cloud Run.

---
ProductPod is built to orchestrate Codex Agents and drive async feature delivery with LLMs and GitHub automation. 🚀