# ai-delivery-framework-gemini

This repository contains a proof of concept for a FastAPI backend that proxies requests to Google's Gemini API and exposes a set of GitHub automation tools via [FastMCP](https://pypi.org/project/modelcontextprotocol/). A minimal front‑end is provided to demonstrate a chat interface.

## Setup

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r app/backend/requirements.txt
   pip install -r project/sample/requirements.txt
   pip install -r requirements-dev.txt  # development tools
   ```
2. **Environment variables**
   - `GEMINI_API_KEY` – API key for Gemini.
   - `GEMINI_MODEL_NAME` – (optional) model version, defaults to `gemini-1.5-flash-latest`.
   - `ALLOWED_ORIGINS` – comma separated origins for CORS (defaults to `*`).

Create a `.env` file in the project root with these variables.

## Running the backend

```bash
uvicorn app.backend.main:app --reload
```

The front-end files in `app/frontend` can be served by any web server. For development you can simply open `index.html` in your browser after starting the backend.

## Testing

Run the unit tests with:

```bash
pytest
```

The tests use FastAPI's `TestClient` to verify that the root endpoint is reachable.

## Sample project

The `project/sample` directory contains an additional FastAPI example with an OpenAPI specification that can be used by Gemini for tool discovery. It is independent of the main backend but demonstrates a larger API surface.
