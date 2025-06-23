import os
import json
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_API_BASE_URL = "https://ai-delivery-framework-production.up.railway.app" # Replace if your GitHub tool server is different
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-1.5-pro-latest"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ProductPod Backend Proxy",
    description="A FastAPI application to securely proxy Gemini API calls and handle GitHub interactions.",
    version="0.1.0"
)

# --- CORS Configuration ---
# WARNING: This allows all origins. Restrict this in a production environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- GitHub Tool Definitions for Gemini ---
# Load the OpenAPI spec for GitHub tools
try:
    with open("../../project/sample/openapi_gemini.json", "r") as f: # Relative path from app/backend/
        openapi_spec = json.load(f)
except FileNotFoundError:
    print("ERROR: openapi_gemini.json not found. Make sure the path is correct.")
    openapi_spec = {"paths": {}} # Fallback to prevent crash

def extract_tool_schema(openapi_spec, path, method="post"):
    """
    Extracts and transforms a single OpenAPI path item into a Gemini FunctionDeclaration-like dictionary.
    """
    path_item = openapi_spec.get("paths", {}).get(path, {}).get(method, {})
    if not path_item:
        return None

    # Extract description from summary or description
    description = path_item.get("summary", path_item.get("description", ""))
    if "x-gpt-action" in path_item and "instructions" in path_item["x-gpt-action"]:
        description = path_item["x-gpt-action"]["instructions"] # Prefer more detailed instructions

    parameters = {}
    required_params = []

    if "requestBody" in path_item:
        request_body_schema = path_item.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
        parameters = request_body_schema.get("properties", {})
        required_params = request_body_schema.get("required", [])

    # Gemini expects 'object' type for parameters block
    return {
        "name": path_item.get("operationId", path.strip("/").replace("/", "_")), # Fallback name
        "description": description,
        "parameters": {
            "type": "object", # Gemini requires this outer type to be 'object'
            "properties": parameters,
            "required": required_params
        }
    }

# Define the tools we want to expose to Gemini from the OpenAPI spec
# These names (e.g., "gitCommit", "sandboxInit", "fetchFiles") will be used by Gemini
# and must match the 'name' in the FunctionDeclaration.
tool_schemas = []

# Tool 1: /git/commit
commit_schema = extract_tool_schema(openapi_spec, "/git/commit")
if commit_schema:
    # Ensure the name matches what Gemini will call
    commit_schema["name"] = "gitCommit" # Or use path_item.get("operationId") if that's what you want Gemini to use
    tool_schemas.append(FunctionDeclaration(**commit_schema))

# Tool 2: /sandbox/init
sandbox_init_schema = extract_tool_schema(openapi_spec, "/sandbox/init")
if sandbox_init_schema:
    sandbox_init_schema["name"] = "sandboxInit"
    tool_schemas.append(FunctionDeclaration(**sandbox_init_schema))

# Tool 3: /system/fetch_files
fetch_files_schema = extract_tool_schema(openapi_spec, "/system/fetch_files")
if fetch_files_schema:
    fetch_files_schema["name"] = "fetchFiles" # This is the name Gemini will use in its function_call
    tool_schemas.append(FunctionDeclaration(**fetch_files_schema))

# Create a Gemini Tool object
GEMINI_TOOLS = [Tool(function_declarations=tool_schemas)] if tool_schemas else None

# --- Security Check ---
if not GEMINI_API_KEY:
    # This check runs at startup.
    # Consider if you want to raise an error here or within the endpoint.
    # Raising here prevents the app from starting if misconfigured.
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")
    # For a real deployment, you might exit or have a more robust config check.
    # For now, we'll let it start but the endpoint will fail.

# --- Pydantic Models for Request/Response ---
class ChatMessage(BaseModel):
    role: str # "user" or "model"
    parts: list[str]

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    # current_tools_output: Optional[list[ToolOutput]] = None # If you were to pass tool output from client

# --- Helper Function to Execute GitHub Tools ---
async def execute_github_tool(tool_name: str, tool_args: dict):
    """
    Executes a specified GitHub tool by making an internal HTTP call.
    """
    endpoint_map = {
        "gitCommit": "/git/commit",
        "sandboxInit": "/sandbox/init",
        "fetchFiles": "/system/fetch_files",
        # Add other tools here if needed
    }

    if tool_name not in endpoint_map:
        return {"error": f"Tool '{tool_name}' is not recognized or supported."}

    api_url = f"{GITHUB_API_BASE_URL}{endpoint_map[tool_name]}"
    # GITHUB_PAT is not directly used by this proxy for auth with the GitHub tool server,
    # as that server handles its own auth. If the tool server required a PAT from this proxy,
    # you would add it to the headers here.
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {os.getenv('GITHUB_PAT')}" # If the target service needs a PAT
    }

    async with httpx.AsyncClient() as client:
        try:
            print(f"Executing tool '{tool_name}' with args: {tool_args} at {api_url}")
            response = await client.post(api_url, json=tool_args, headers=headers, timeout=30.0)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error calling tool {tool_name}: {e.response.status_code} - {e.response.text}")
            return {"error": f"Error calling tool {tool_name}: {e.response.status_code}", "details": e.response.text}
        except httpx.RequestError as e:
            print(f"Request error calling tool {tool_name}: {e}")
            return {"error": f"Request error calling tool {tool_name}: {str(e)}"}
        except json.JSONDecodeError:
            print(f"JSON decode error for tool {tool_name} response: {response.text}")
            return {"error": f"Failed to decode JSON response from tool {tool_name}"}

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY is not set. The /gemini/chat endpoint will not function.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            print("Gemini API configured successfully.")
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")

    if not GEMINI_TOOLS:
        print("WARNING: No Gemini tools were loaded. Tool functionality will be disabled.")
    else:
        print(f"Gemini tools loaded: {[tool.name for tool in tool_schemas]}")


@app.post("/gemini/chat")
async def gemini_chat_proxy(chat_request: ChatRequest, request: Request):
    """
    Proxies chat requests to the Gemini API, handling tool calls internally.
    """
    if not GEMINI_API_KEY:
        # Remind user to set the API key in their environment.
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on the server. Please set the environment variable.")

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            tools=GEMINI_TOOLS if GEMINI_TOOLS else None
        )

        # Convert Pydantic models to dicts for Gemini SDK
        history = []
        for msg in chat_request.messages:
            history.append({"role": msg.role, "parts": msg.parts})

        print(f"Sending to Gemini with history: {history}")
        if GEMINI_TOOLS:
            print(f"And tools: {[fc.name for fc in GEMINI_TOOLS[0].function_declarations]}")


        # Initial call to Gemini
        response = await model.generate_content_async(history)
        candidate = response.candidates[0] # Using the first candidate

        MAX_TOOL_ITERATIONS = 5 # Prevent infinite loops
        current_iteration = 0

        while candidate.finish_reason == "TOOL_CODE" and current_iteration < MAX_TOOL_ITERATIONS:
            print("Gemini suggests a tool call.")
            if not candidate.content.parts:
                 # This can happen if Gemini ONLY returns a tool call without any preceding text.
                print("No content parts in candidate, only tool call.")
            else:
                 # Add Gemini's response part (if any) before the tool call to history
                history.append({"role": "model", "parts": [part.text for part in candidate.content.parts if hasattr(part, 'text')]})


            function_calls = [part.function_call for part in candidate.content.parts if hasattr(part, 'function_call')]

            if not function_calls:
                print("Error: Gemini indicated tool use but no function_call found.")
                return {"text": "Error in processing tool call from Gemini."} # Or try to return text if available

            tool_results = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) # Convert FunctionCallArgs to dict
                print(f"Attempting to execute tool: {tool_name} with args: {tool_args}")

                # Execute the tool
                tool_output_content = await execute_github_tool(tool_name, tool_args)
                print(f"Tool '{tool_name}' executed. Output: {tool_output_content}")

                tool_results.append({
                    "executable_part": { # For Gemini API, this structure is slightly different than just `tool_code`
                        "function_response": {
                            "name": tool_name,
                            "response": tool_output_content # Gemini expects the actual content here
                        }
                    }
                })

            # Add the function call results to the history for the next turn
            # The structure for sending back tool results to Gemini:
            history.append({
                "role": "user", # Or "tool" or "function" depending on exact API spec for this part
                                # For Gemini API, the role is "user" for function responses
                "parts": tool_results
            })

            print(f"Sending tool results back to Gemini. Updated history: {history}")
            # Make a follow-up call to Gemini with the tool's output
            response = await model.generate_content_async(history)
            candidate = response.candidates[0]
            current_iteration += 1

        if current_iteration >= MAX_TOOL_ITERATIONS:
            print("Max tool iterations reached.")
            return {"text": "Max tool iterations reached. Please try a simpler request or check tool behavior."}

        # Once Gemini responds with text (or no more tool calls)
        if candidate.content and candidate.content.parts:
            final_text_response = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
            print(f"Final response from Gemini: {final_text_response}")
            return {"text": final_text_response}
        elif function_calls and not candidate.content.parts: # Should not happen if loop exited correctly
             # If the loop exited due to max iterations but the last response was still a tool call
            return {"text": "Model is still suggesting tool calls after max iterations.", "function_call_suggestion": function_calls[0].name}
        else:
            print("Gemini response did not contain expected text content.")
            return {"text": "No text content received from Gemini."}

    except Exception as e:
        print(f"Error in /gemini/chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Root Endpoint for Health Check ---
@app.get("/")
async def root():
    return {"message": "ProductPod Backend Proxy is running."}

# --- To run this app (from the `app/backend` directory): ---
# uvicorn main:app --reload --port 8000
# Ensure you have a .env file in app/backend/ with your GEMINI_API_KEY
#
# Example .env content:
# GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_KEY
# GITHUB_PAT=YOUR_GITHUB_PAT (optional, for tools that might need it indirectly)
#
# Reminder: For deployment, set these environment variables directly in your hosting environment.
# The GITHUB_PAT is not directly used by this proxy for its own authentication to the GitHub tool server.
# The tool server (e.g., ai-delivery-framework-production.up.railway.app) handles its own authentication.
# If that server required a PAT from this proxy for its calls, you would add it to the `execute_github_tool` headers.
#
# The tools from openapi_gemini.json are:
# - /git/commit (mapped to 'gitCommit')
# - /sandbox/init (mapped to 'sandboxInit')
# - /system/fetch_files (mapped to 'fetchFiles')
# These names ('gitCommit', 'sandboxInit', 'fetchFiles') are what Gemini will use in `function_call.name`.
# The `execute_github_tool` function then maps these names back to the actual API paths.
#
# IMPORTANT: The path to `openapi_gemini.json` is relative: `../../project/sample/openapi_gemini.json`.
# This means it expects to be run from the `app/backend` directory.
# If you run uvicorn from the root of the repository, this path will be incorrect.
# Uvicorn command: `uvicorn app.backend.main:app --reload --port 8000` (run from repo root)
# Or: `cd app/backend && uvicorn main:app --reload --port 8000`
# The provided uvicorn command in the plan `uvicorn app.backend.main:app --reload --port 8000` assumes running from the repo root.
# I will adjust the openapi_spec path to be relative to the repo root for consistency.

# Corrected path for openapi_spec assuming uvicorn is run from repo root
try:
    # Path relative to the root of the repository
    openapi_spec_path = "project/sample/openapi_gemini.json"
    with open(openapi_spec_path, "r") as f:
        openapi_spec = json.load(f)
    print(f"Successfully loaded OpenAPI spec from: {openapi_spec_path}")
except FileNotFoundError:
    print(f"ERROR: openapi_gemini.json not found at {openapi_spec_path}. Please ensure the path is correct relative to the repository root.")
    openapi_spec = {"paths": {}} # Fallback
except Exception as e:
    print(f"An error occurred loading openapi_gemini.json: {e}")
    openapi_spec = {"paths": {}} # Fallback

# Re-initialize tools with corrected path logic
tool_schemas = []
commit_schema = extract_tool_schema(openapi_spec, "/git/commit")
if commit_schema:
    commit_schema["name"] = "gitCommit"
    tool_schemas.append(FunctionDeclaration(**commit_schema))

sandbox_init_schema = extract_tool_schema(openapi_spec, "/sandbox/init")
if sandbox_init_schema:
    sandbox_init_schema["name"] = "sandboxInit"
    tool_schemas.append(FunctionDeclaration(**sandbox_init_schema))

fetch_files_schema = extract_tool_schema(openapi_spec, "/system/fetch_files")
if fetch_files_schema:
    fetch_files_schema["name"] = "fetchFiles"
    tool_schemas.append(FunctionDeclaration(**fetch_files_schema))

GEMINI_TOOLS = [Tool(function_declarations=tool_schemas)] if tool_schemas else None
# Update startup log for tools
@app.on_event("startup")
async def startup_event_updated(): # Renamed to avoid redefinition error if cells run multiple times
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY is not set. The /gemini/chat endpoint will not function.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            print("Gemini API configured successfully.")
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")

    if not GEMINI_TOOLS:
        print("WARNING: No Gemini tools were loaded. Tool functionality will be disabled.")
    else:
        print(f"Gemini tools loaded: {[tool.name for tool_declaration in GEMINI_TOOLS[0].function_declarations for tool in [tool_declaration]] if GEMINI_TOOLS and GEMINI_TOOLS[0].function_declarations else 'None'}")

# Replace the old startup event with the updated one
app.router.on_startup.pop() # Remove the previous one if it was added
app.router.on_startup.append(startup_event_updated)

print("Backend main.py script defined.")
