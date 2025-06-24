import os
import json
import copy # For deepcopy
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool as GeminiSdkTool
from google.protobuf.struct_pb2 import Value as ProtoValue, Struct as ProtoStruct, ListValue as ProtoListValue

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx

from pydantic import BaseModel, Field 

from mcp.server.fastmcp import FastMCP, Context
from mcp.shared.context import RequestContext
from mcp.types import JSONRPCMessage, RequestId 

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

dotenv_path = Path(__file__).resolve().parent / ".env" 
load_dotenv(dotenv_path=dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_API_BASE_URL = "https://ai-delivery-framework-production.up.railway.app"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

app = FastAPI(
    title="ProductPod Backend Proxy", 
    description="A FastAPI application to securely proxy Gemini API calls and handle GitHub interactions, with MCP tools.",
    version="0.3.17" # Incremented version
)

mcp_app = FastMCP(
    title="ProductPod MCP Tool Server", 
    description="Exposes ProductPod's GitHub tools via MCP."
)
app.mount("/mcp", mcp_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")

async def _execute_actual_github_api_call(api_path: str, tool_args: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    api_url = f"{GITHUB_API_BASE_URL}{api_path}"
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            print(f"[_execute_actual_github_api_call] Executing API call for tool '{tool_name}' with args: {tool_args} at {api_url}")
            response = await client.post(api_url, json=tool_args, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Error calling GitHub API for {tool_name}: {e.response.status_code}", "details": e.response.text}
        except httpx.RequestError as e:
            return {"error": f"Request error calling GitHub API for {tool_name}: {str(e)}"}
        except json.JSONDecodeError:
            resp_text = response.text if 'response' in locals() else 'No response object'
            return {"error": f"Failed to decode JSON response from GitHub API for {tool_name}"}

class FetchFilesInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name (e.g., 'owner/repo')")
    branch: Optional[str] = Field(default="main", description="Git branch to fetch from (default is 'main')")
    paths: List[str] = Field(description="List of full file paths to fetch from the repository root.")

class FetchFilesOutput(BaseModel): 
    files: Dict[str, str] = Field(description="Mapping of file paths to content")
    error: Optional[str] = Field(default=None, description="Error message if fetching failed")

@mcp_app.tool(name="fetchFiles") 
async def fetch_files_mcp_tool(input: FetchFilesInput, context: Context) -> FetchFilesOutput: 
    """Fetches one or more files from a specific GitHub repository and branch."""
    await context.info(f"[fetch_files_mcp_tool] Called for repo: {input.repo_name} paths: {input.paths}")
    api_args = {"repo_name": input.repo_name, "branch": input.branch, "paths": input.paths, "mode": "batch"}
    tool_output_content = await _execute_actual_github_api_call("/system/fetch_files", api_args, "fetchFiles")
    if "error" in tool_output_content:
        await context.error(f"[fetch_files_mcp_tool] Error: {tool_output_content.get('details', tool_output_content['error'])}")
        return FetchFilesOutput(files={}, error=str(tool_output_content.get('details', tool_output_content['error'])))
    return FetchFilesOutput(**tool_output_content)

class GitCommitInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name (e.g., 'owner/repo')")
    branch: str = Field(description="Git branch to commit to (e.g., 'main', 'develop')")
    message: str = Field(description="The commit message.")
    paths: Optional[List[str]] = Field(default=None, description="Optional list of specific file paths to add and commit. If None, commits all staged changes.")

class GitCommitOutput(BaseModel):
    commit_url: Optional[str] = Field(default=None, description="URL of the new commit if successful.")
    status: str = Field(description="Status of the commit operation (e.g., 'success', 'failure').")
    error: Optional[str] = Field(default=None, description="Error message if the commit failed.")

@mcp_app.tool(name="gitCommit")
async def git_commit_mcp_tool(input: GitCommitInput, context: Context) -> GitCommitOutput:
    """Commits changes to a specified GitHub repository and branch."""
    await context.info(f"[git_commit_mcp_tool] Called for repo: {input.repo_name}, branch: {input.branch}")
    api_args = input.model_dump()
    if api_args.get("paths") is None: api_args.pop("paths", None) 
    tool_output_content = await _execute_actual_github_api_call("/git/commit", api_args, "gitCommit")
    if "error" in tool_output_content:
        await context.error(f"[git_commit_mcp_tool] Error: {tool_output_content.get('details', tool_output_content['error'])}")
        return GitCommitOutput(status="failure", error=str(tool_output_content.get('details', tool_output_content['error'])))
    return GitCommitOutput(commit_url=tool_output_content.get("commit_url"), status=tool_output_content.get("status", "success"), **tool_output_content)

class SandboxInitInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name (e.g., 'owner/repo')")
    mode: str = Field(description="Mode of operation: 'branch' or 'project'.")
    reuse_token: Optional[str] = Field(default=None); force_new: Optional[bool] = Field(default=False)
    branch: Optional[str] = Field(default=None); project_name: Optional[str] = Field(default=None)
    project_description: Optional[str] = Field(default=None)

class SandboxInitOutput(BaseModel):
    branch: Optional[str] = Field(default=None); repo_name: Optional[str] = Field(default=None)
    reuse_token: Optional[str] = Field(default=None); message: str
    error: Optional[str] = Field(default=None)

@mcp_app.tool(name="sandboxInit")
async def sandbox_init_mcp_tool(input: SandboxInitInput, context: Context) -> SandboxInitOutput:
    """Initializes a sandbox environment."""
    await context.info(f"[sandbox_init_mcp_tool] Called for repo: {input.repo_name}, mode: {input.mode}")
    api_args = input.model_dump(exclude_none=True) 
    tool_output_content = await _execute_actual_github_api_call("/sandbox/init", api_args, "sandboxInit")
    if "error" in tool_output_content:
        await context.error(f"[sandbox_init_mcp_tool] Error: {tool_output_content.get('details', tool_output_content['error'])}")
        return SandboxInitOutput(message="Initialization failed", error=str(tool_output_content.get('details', tool_output_content['error'])))
    return SandboxInitOutput(**tool_output_content)

def _recursive_convert_gemini_struct_to_dict(value: Any) -> Any:
    if isinstance(value, ProtoStruct): # Handle top-level Struct (like fc.args) or nested Structs
         return {key: _recursive_convert_gemini_struct_to_dict(val) for key, val in value.fields.items()}
    elif isinstance(value, ProtoListValue): # Handle ListValue
         return [_recursive_convert_gemini_struct_to_dict(item) for item in value.values]
    elif isinstance(value, ProtoValue): # Handle Value wrappers
        kind = value.WhichOneof('kind')
        if kind == 'string_value': return value.string_value
        if kind == 'number_value': return value.number_value
        if kind == 'bool_value': return value.bool_value
        if kind == 'null_value': return None
        if kind == 'struct_value': return _recursive_convert_gemini_struct_to_dict(value.struct_value) 
        if kind == 'list_value': return _recursive_convert_gemini_struct_to_dict(value.list_value) 
        return value # Should not happen for a Value message if kind is one of above
    # Fallbacks for already converted Python types (e.g. if part of the structure was already dict/list)
    elif isinstance(value, dict): 
        return {k: _recursive_convert_gemini_struct_to_dict(v) for k, v in value.items()}
    elif isinstance(value, list): 
        return [_recursive_convert_gemini_struct_to_dict(v) for v in value]
    return value # Primitives (str, int, float, bool, None already handled by ProtoValue)

def _resolve_refs_and_simplify(schema_node: Any, definitions: Dict[str, Any], tool_name: str) -> Any:
    if isinstance(schema_node, list): return [_resolve_refs_and_simplify(item, definitions, tool_name) for item in schema_node]
    if not isinstance(schema_node, dict): return schema_node
    node_copy = schema_node.copy()
    if '$ref' in node_copy:
        ref_path = node_copy['$ref']; def_key = ref_path.split('/')[-1]
        if def_key in definitions:
            resolved_def = _resolve_refs_and_simplify(definitions[def_key], definitions, tool_name)
            merged_def = {**resolved_def} 
            for k, v_ref_node in node_copy.items():
                if k != '$ref' and k not in merged_def: merged_def[k] = v_ref_node
            return merged_def
        else: return {"type": "string", "description": f"Unresolved reference: {ref_path}"}
    if 'anyOf' in node_copy:
        primary_type_schema = next((opt for opt in node_copy['anyOf'] if isinstance(opt, dict) and opt.get('type') != 'null'), None)
        if primary_type_schema:
            new_node = primary_type_schema.copy()
            for key, value in node_copy.items():
                if key not in ['anyOf', 'type'] and key not in new_node : new_node[key] = value
            return _resolve_refs_and_simplify(new_node, definitions, tool_name)
        else: return {"type": "string", "description": "Simplified anyOf field"}
    processed_node = {}
    for key, value in node_copy.items():
        if key in ['$defs', 'definitions', 'title', 'default']: continue 
        processed_node[key] = _resolve_refs_and_simplify(value, definitions, tool_name)
    return processed_node

async def get_gemini_tools_from_fastmcp(mcp_server: FastMCP) -> List[GeminiSdkTool]:
    gemini_function_declarations = []
    try:
        mcp_tools = await mcp_server.list_tools() 
        if not mcp_tools: return []
        for mcp_tool in mcp_tools: 
            tool_name = mcp_tool.name; tool_description = mcp_tool.description or ""
            original_schema = copy.deepcopy(mcp_tool.inputSchema) 
            definitions = original_schema.pop('$defs', original_schema.pop('definitions', {}))
            if 'title' in original_schema: del original_schema['title']
            if 'default' in original_schema: del original_schema['default']
            final_parameters = _resolve_refs_and_simplify(original_schema, definitions, tool_name)
            if not (isinstance(final_parameters, dict) and final_parameters.get("type") == "object"):
                final_parameters = {"type": "object", "properties": {}, "required": []}
            if "properties" not in final_parameters: final_parameters["properties"] = {}
            if "required" not in final_parameters: final_parameters["required"] = []
            elif not isinstance(final_parameters["required"], list): final_parameters["required"] = []
            
            print(f"--- [get_gemini_tools_from_fastmcp] Preparing tool for Gemini: {tool_name} ---")
            print(f"Description: {tool_description}")
            print(f"Parameters (final for Gemini): {json.dumps(final_parameters, indent=2)}")
            function_declaration = FunctionDeclaration(name=tool_name, description=tool_description, parameters=final_parameters)
            gemini_function_declarations.append(function_declaration)
    except Exception as e:
        print(f"[get_gemini_tools_from_fastmcp] Error: {e}"); import traceback; traceback.print_exc()
        return [] 
    if gemini_function_declarations: return [GeminiSdkTool(function_declarations=gemini_function_declarations)]
    return []

GEMINI_TOOLS_FOR_SDK: List[GeminiSdkTool] = [] 

@app.on_event("startup")
async def startup_event_mcp(): 
    global GEMINI_TOOLS_FOR_SDK
    if not GEMINI_API_KEY: print("WARNING: GEMINI_API_KEY is not set.")
    else:
        try: genai.configure(api_key=GEMINI_API_KEY); print("Gemini API configured successfully.")
        except Exception as e: print(f"Error configuring Gemini API: {e}")
    GEMINI_TOOLS_FOR_SDK = await get_gemini_tools_from_fastmcp(mcp_app)
    if not GEMINI_TOOLS_FOR_SDK: print("WARNING: No Gemini tools derived from FastMCP.")
    else: print(f"Gemini tools derived from FastMCP: {[fd.name for tool in GEMINI_TOOLS_FOR_SDK for fd in tool.function_declarations]}")

@app.post("/gemini/chat")
async def gemini_chat_proxy_mcp(request: Request): 
    print("[/gemini/chat] Received request.")
    if not GEMINI_API_KEY: raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on the server.")
    try:
        request_data = await request.json()
        gemini_history = []
        if "parts" in request_data and isinstance(request_data["parts"], list):
            for part_data in request_data["parts"]:
                if part_data.get("type") == "chat" and "role" in part_data and "parts" in part_data:
                    text_sub_parts = [p.get("text") for p in part_data["parts"] if p.get("type") == "text" and "text" in p]
                    if text_sub_parts: gemini_history.append({"role": part_data["role"], "parts": [{"text": t} for t in text_sub_parts]})
        if not gemini_history: raise HTTPException(status_code=400, detail="Invalid or empty chat history in request.")

        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, tools=GEMINI_TOOLS_FOR_SDK)
        MAX_TOOL_ITERATIONS = 5; current_iteration = 0

        while current_iteration < MAX_TOOL_ITERATIONS:
            response = await model.generate_content_async(gemini_history)
            candidate = response.candidates[0]
            print(f"[/gemini/chat] Gemini raw candidate object: {candidate}") 
            print(f"[/gemini/chat] Gemini response candidate: Finish Reason: {candidate.finish_reason}")
            
            function_calls = []
            model_turn_parts_for_history = [] 

            if candidate.content and candidate.content.parts:
                print(f"[/gemini/chat] Gemini response parts: {candidate.content.parts}")
                for part in candidate.content.parts: 
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call) 
                        fc_for_history = {
                            "name": part.function_call.name,
                            "args": _recursive_convert_gemini_struct_to_dict(part.function_call.args) 
                        }
                        model_turn_parts_for_history.append({"function_call": fc_for_history})
                    elif hasattr(part, 'text'):
                         model_turn_parts_for_history.append({"text": part.text})
            
            if model_turn_parts_for_history: 
                gemini_history.append({"role": "model", "parts": model_turn_parts_for_history})

            if function_calls: 
                print(f"[/gemini/chat] Function call(s) detected in parts: {function_calls}")
                tool_results_for_gemini = []
                for fc in function_calls: 
                    tool_name = fc.name
                    tool_args_as_dict = _recursive_convert_gemini_struct_to_dict(fc.args) if fc.args else {}
                    
                    print(f"[/gemini/chat] Processing tool call: {tool_name}, Args after _recursive_convert_gemini_struct_to_dict: {tool_args_as_dict}")

                    actual_tool_args_for_pydantic = tool_args_as_dict.get("input", tool_args_as_dict)
                    if not isinstance(actual_tool_args_for_pydantic, dict):
                        print(f"[/gemini/chat] Warning: Extracted 'input' args for Pydantic model of tool {tool_name} is not a dict (or is None). Value: {actual_tool_args_for_pydantic}. Using empty dict for Pydantic if tool expects args.")
                        actual_tool_args_for_pydantic = {} 
                    
                    tool_mcp_instance = None
                    if hasattr(mcp_app, '_tool_manager') and mcp_app._tool_manager: tool_mcp_instance = mcp_app._tool_manager.get_tool(tool_name)
                    
                    if tool_mcp_instance and hasattr(tool_mcp_instance, 'func'):
                        try:
                            pydantic_input_model = tool_mcp_instance.model
                            input_instance = None
                            if pydantic_input_model:
                                print(f"[/gemini/chat] Attempting to instantiate Pydantic model '{pydantic_input_model.__name__}' with args: {actual_tool_args_for_pydantic}")
                                input_instance = pydantic_input_model(**actual_tool_args_for_pydantic)
                            elif actual_tool_args_for_pydantic: 
                                print(f"[/gemini/chat] Warning: Tool {tool_name} has no Pydantic input model, but args were provided by Gemini: {actual_tool_args_for_pydantic}. These args will be ignored.")
                                                        
                            fake_req_ctx = RequestContext(request_id=RequestId.new(), client_id="gemini_internal_caller", session_id=None)
                            tool_mgr_inst = mcp_app._tool_manager if hasattr(mcp_app, '_tool_manager') else None
                            minimal_ctx = Context(request_context=fake_req_ctx, tool_manager=tool_mgr_inst)
                            
                            if input_instance is not None: tool_output = await tool_mcp_instance.func(input_instance, context=minimal_ctx)
                            elif not pydantic_input_model : tool_output = await tool_mcp_instance.func(context=minimal_ctx) 
                            else: raise ValueError(f"Pydantic input_instance for tool {tool_name} is None, but model exists. Args were: {actual_tool_args_for_pydantic}")
                            
                            if isinstance(tool_output, BaseModel): tool_output_content = tool_output.model_dump() 
                            elif isinstance(tool_output, dict): tool_output_content = tool_output
                            else: tool_output_content = {"error": "Tool returned unexpected type", "type": str(type(tool_output))}
                        except Exception as e:
                            print(f"[/gemini/chat] Error executing FastMCP tool '{tool_name}' directly: {e}"); import traceback; traceback.print_exc()
                            tool_output_content = {"error": f"Failed to execute tool {tool_name}: {str(e)}"}
                    else:
                        print(f"[/gemini/chat] Error: FastMCP tool '{tool_name}' not found or not callable (mcp_instance: {tool_mcp_instance}).")
                        tool_output_content = {"error": f"Tool {tool_name} not found."}
                    
                    print(f"[/gemini/chat] Tool '{tool_name}' executed. Output for Gemini: {json.dumps(tool_output_content)}")
                    tool_results_for_gemini.append({"function_response": {"name": tool_name, "response": tool_output_content}})
                
                gemini_history.append({"role": "user", "parts": tool_results_for_gemini}) 
                current_iteration += 1; continue 
            
            print(f"[/gemini/chat] No function call in this turn or finish_reason ({candidate.finish_reason}) is not actionable for tools. Treating as final text response.")
            final_text_response = ""
            if candidate.content and candidate.content.parts: 
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                if text_parts: final_text_response = "".join(text_parts)
            
            print(f"[/gemini/chat] Final text response from Gemini: '{final_text_response}'")
            response_to_frontend = {"parts": [{"type": "chat", "role": "model", "parts": [{"type": "text", "text": final_text_response}]}]}
            return JSONResponse(content=response_to_frontend)
        
        print("[/gemini/chat] Max tool iterations reached.")
        timeout_response = {"parts": [{"type": "chat", "role": "model", "parts": [{"type": "text", "text": "Max tool iterations reached."}]}]}
        return JSONResponse(content=timeout_response)
    except json.JSONDecodeError as e:
        print(f"[/gemini/chat] JSONDecodeError: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in request: {str(e)}")
    except HTTPException as e: raise e
    except Exception as e:
        print(f"[/gemini/chat] Generic Error: {e}"); import traceback; traceback.print_exc()
        error_response = {"parts": [{"type": "chat", "role": "model", "parts": [{"type": "text", "text": f"An error occurred: {str(e)}"}]}]}
        return JSONResponse(content=error_response, status_code=500)

@app.get("/")
async def root_mcp_main(): 
    return {"message": "ProductPod Backend Proxy (with FastMCP tools) is running."}

if hasattr(app, 'router') and app.router.on_startup:
    app.router.on_startup = [e for e in app.router.on_startup if getattr(e, '__name__', '') not in ('startup_event_mcp_old', 'startup_event')] 
app.router.on_startup.append(startup_event_mcp)

print("Backend main.py script (FastMCP integration version) defined.")
