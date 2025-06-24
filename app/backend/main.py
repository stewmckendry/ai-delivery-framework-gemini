# app/backend/main.py

import os
import json
import copy
from pathlib import Path # Added for Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import Tool as GeminiSdkTool, FunctionDeclaration
from google.protobuf.struct_pb2 import Value as ProtoValue, Struct as ProtoStruct, ListValue as ProtoListValue # For Gemini

from pydantic import BaseModel # For tool_output.model_dump if needed by Gemini response

from mcp.server.fastmcp import FastMCP, Context
from mcp.shared.context import RequestContext # For fake_req_ctx
from mcp.types import RequestId # For fake_req_ctx

# Import tools and Pydantic models from the new github_tools.py
# Note: Pydantic models are defined in github_tools.py and used by the tools there.
# They don't need to be explicitly imported here unless main.py itself uses them directly.
from .github_tools import (
    fetch_files_mcp_tool,
    list_files_mcp_tool,
    search_files_in_repo_mcp_tool,
    git_commit_mcp_tool,
    preview_changes_mcp_tool,
    sandbox_init_mcp_tool
    # _recursive_convert_gemini_struct_to_dict # This is general Gemini utility, keep here or move to general utils
)

# Attempt to find .env file by going up two levels from this file's location (app/backend/main.py -> app/ -> .env)
# This is a common pattern for projects where .env is at the project root.
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if not dotenv_path.exists():
    # Fallback for cases where structure might be different (e.g. running tests, or if main is nested deeper)
    print(f"Warning: .env file not found at {dotenv_path}. Trying another common location: project root from current file's grandparent's parent.")
    dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not dotenv_path.exists():
         print(f"Warning: .env file not found at {dotenv_path} either. Environment variables might not be loaded if not set externally.")

load_dotenv(dotenv_path=dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

app = FastAPI(
    title="ProductPod Backend Proxy", 
    description="A FastAPI application to securely proxy Gemini API calls and handle GitHub interactions, with MCP tools.",
    version="0.3.20" # Incremented version for refactoring
)

mcp_app = FastMCP(
    title="ProductPod MCP Tool Server", 
    description="Exposes ProductPod's GitHub tools via MCP."
)

# Register tools from github_tools.py
mcp_app.tool(name="fetchFiles")(fetch_files_mcp_tool)
mcp_app.tool(name="listFiles")(list_files_mcp_tool)
mcp_app.tool(name="searchFilesInRepo")(search_files_in_repo_mcp_tool)
mcp_app.tool(name="gitCommit")(git_commit_mcp_tool) # This is the multi-file commit tool
mcp_app.tool(name="previewChanges")(preview_changes_mcp_tool)
mcp_app.tool(name="sandboxInit")(sandbox_init_mcp_tool)

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


# Utility for Gemini function call argument conversion (remains in main.py as it's Gemini specific)
from typing import Any # Ensure Any is imported for the function signature
def _recursive_convert_gemini_struct_to_dict(value: Any) -> Any:
    if isinstance(value, ProtoStruct):
         return {key: _recursive_convert_gemini_struct_to_dict(val) for key, val in value.fields.items()}
    elif isinstance(value, ProtoListValue):
         return [_recursive_convert_gemini_struct_to_dict(item) for item in value.values]
    elif isinstance(value, ProtoValue):
        kind = value.WhichOneof('kind')
        if kind == 'string_value': return value.string_value
        if kind == 'number_value': return value.number_value
        if kind == 'bool_value': return value.bool_value
        if kind == 'null_value': return None
        if kind == 'struct_value': return _recursive_convert_gemini_struct_to_dict(value.struct_value) 
        if kind == 'list_value': return _recursive_convert_gemini_struct_to_dict(value.list_value) 
        return value # Should not happen for a Value message if kind is one of above
    elif isinstance(value, dict): 
        return {k: _recursive_convert_gemini_struct_to_dict(v) for k, v in value.items()}
    elif isinstance(value, list): 
        return [_recursive_convert_gemini_struct_to_dict(v) for v in value]
    return value


async def get_gemini_tools_from_fastmcp(mcp_server: FastMCP) -> List[GeminiSdkTool]:
    gemini_function_declarations = []
    try:
        mcp_tools_list = await mcp_server.list_tools() # Renamed to avoid conflict with GeminiSdkTool
        if not mcp_tools_list:
            print("[get_gemini_tools_from_fastmcp] No MCP tools found.")
            return []
        for mcp_tool_item in mcp_tools_list: # Renamed to avoid conflict
            tool_name = mcp_tool_item.name
            tool_description = mcp_tool_item.description or f"Tool: {tool_name}"
            parameters_schema = copy.deepcopy(mcp_tool_item.inputSchema)
            if not isinstance(parameters_schema, dict) or parameters_schema.get("type") != "object":
                # If schema is not a Pydantic model producing object type (e.g. simple type directly)
                # Gemini might still need it wrapped. For now, assume valid object schema from Pydantic.
                print(f"[get_gemini_tools_from_fastmcp] Warning: Schema for tool '{tool_name}' is not a direct object schema or is None. Using empty properties.")
                parameters_schema = {"type": "object", "properties": {}} if parameters_schema is None else parameters_schema

            if 'title' in parameters_schema: del parameters_schema['title'] # Gemini uses tool name/desc

            function_declaration = FunctionDeclaration(
                name=tool_name,
                description=tool_description,
                parameters=parameters_schema
            )
            gemini_function_declarations.append(function_declaration)
    except Exception as e:
        print(f"[get_gemini_tools_from_fastmcp] Error processing MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return []

    if gemini_function_declarations:
        return [GeminiSdkTool(function_declarations=gemini_function_declarations)]
    print("[get_gemini_tools_from_fastmcp] No function declarations were generated.")
    return []

GEMINI_TOOLS_FOR_SDK: List[GeminiSdkTool] = [] 

@app.on_event("startup")
async def startup_event_mcp(): 
    global GEMINI_TOOLS_FOR_SDK
    if not GEMINI_API_KEY: print("WARNING: GEMINI_API_KEY is not set.")
    else:
        try: genai.configure(api_key=GEMINI_API_KEY); print("Gemini API configured successfully.")
        except Exception as e: print(f"Error configuring Gemini API: {e}")

    print("Fetching tools from MCP for Gemini SDK...")
    GEMINI_TOOLS_FOR_SDK = await get_gemini_tools_from_fastmcp(mcp_app)
    if not GEMINI_TOOLS_FOR_SDK: print("WARNING: No Gemini tools derived from FastMCP.")
    else: print(f"Gemini tools derived from FastMCP: {[fd.name for tool in GEMINI_TOOLS_FOR_SDK for fd in tool.function_declarations]}")


@app.post("/gemini/chat")
async def gemini_chat_proxy_mcp(request: Request): 
    if not GEMINI_API_KEY: raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on the server.")

    # THIS IS A CRITICAL PLACEHOLDER - Replace with actual user authentication and token retrieval
    USER_GITHUB_TOKEN_FOR_TESTING = os.getenv("USER_GITHUB_TOKEN_FOR_TESTING")
    PRODUCT_POD_USER_ID_FOR_TESTING = os.getenv("PRODUCT_POD_USER_ID_FOR_TESTING", "test_user_main_py")
    # END CRITICAL PLACEHOLDER

    try:
        request_data = await request.json()
        gemini_history = request_data.get("parts", [])
        if not gemini_history: raise HTTPException(status_code=400, detail="Invalid or empty chat history in request.")

        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, tools=GEMINI_TOOLS_FOR_SDK)
        MAX_TOOL_ITERATIONS = 10; current_iteration = 0

        while current_iteration < MAX_TOOL_ITERATIONS:
            response = await model.generate_content_async(gemini_history)
            candidate = response.candidates[0]
            function_calls = []
            model_turn_parts_for_history = [] 

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts: 
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call) 
                        fc_for_history = {"name": part.function_call.name, "args": _recursive_convert_gemini_struct_to_dict(part.function_call.args)}
                        model_turn_parts_for_history.append({"function_call": fc_for_history})
                    elif hasattr(part, 'text'):
                         model_turn_parts_for_history.append({"text": part.text})
            if model_turn_parts_for_history: 
                gemini_history.append({"role": "model", "parts": model_turn_parts_for_history})

            if function_calls: 
                tool_results_for_gemini = []
                for fc in function_calls: 
                    tool_name = fc.name
                    tool_args_as_dict = _recursive_convert_gemini_struct_to_dict(fc.args) if fc.args else {}
                    actual_tool_args_for_pydantic = tool_args_as_dict.get("input", tool_args_as_dict)
                    if not isinstance(actual_tool_args_for_pydantic, dict): actual_tool_args_for_pydantic = {}
                    
                    tool_mcp_instance = mcp_app._tool_manager.get_tool(tool_name) if hasattr(mcp_app, '_tool_manager') else None
                    
                    if tool_mcp_instance and hasattr(tool_mcp_instance, 'func'):
                        try:
                            pydantic_input_model = tool_mcp_instance.model # This is the Pydantic model class
                            input_instance = None
                            if pydantic_input_model:
                                input_instance = pydantic_input_model(**actual_tool_args_for_pydantic)
                            elif actual_tool_args_for_pydantic : # Args provided but no model
                                 print(f"Warning: Tool {tool_name} received args but has no input model. Args ignored: {actual_tool_args_for_pydantic}")
                                                        
                            # Prepare context for the tool call, injecting user-specific tokens
                            current_request_context_dict = {
                                "client_id": "gemini_chat_proxy",
                                "request_id": RequestId.new(),
                                "user_github_token": USER_GITHUB_TOKEN_FOR_TESTING, # MUST BE REPLACED by actual user token
                                "product_pod_user_id": PRODUCT_POD_USER_ID_FOR_TESTING # MUST BE REPLACED
                            }
                            if not USER_GITHUB_TOKEN_FOR_TESTING:
                                print(f"WARNING: USER_GITHUB_TOKEN_FOR_TESTING not set. Tool '{tool_name}' might fail if it needs GitHub auth.")

                            fake_req_ctx = RequestContext(**current_request_context_dict)
                            # Ensure mcp_app._tool_manager is valid if used, or pass None
                            tool_manager_for_ctx = mcp_app._tool_manager if hasattr(mcp_app, '_tool_manager') else None
                            minimal_ctx = Context(request_context=fake_req_ctx, tool_manager=tool_manager_for_ctx)
                            
                            if input_instance is not None:
                                tool_output = await tool_mcp_instance.func(input_instance, context=minimal_ctx)
                            elif not pydantic_input_model : # Tool takes no arguments other than context
                                tool_output = await tool_mcp_instance.func(context=minimal_ctx)
                            else: # Has a model, but input_instance is None (should not happen if model exists and args were parsed)
                                raise ValueError(f"Tool {tool_name} has Pydantic model but input_instance is None. Args: {actual_tool_args_for_pydantic}")
                            
                            # Ensure tool_output is a Pydantic model before calling model_dump
                            if isinstance(tool_output, BaseModel):
                                tool_output_content = tool_output.model_dump(exclude_none=True)
                            elif isinstance(tool_output, dict): # Already a dict
                                tool_output_content = tool_output
                            else: # Should not happen if tools return Pydantic models or dicts
                                tool_output_content = {"error": "Tool returned unexpected type", "type": str(type(tool_output))}
                        except Exception as e:
                            import traceback
                            print(f"[/gemini/chat] Error executing FastMCP tool '{tool_name}': {e}"); traceback.print_exc()
                            tool_output_content = {"error": f"Failed to execute tool {tool_name}: {str(e)}"}
                    else:
                        tool_output_content = {"error": f"Tool {tool_name} not found or not callable."}
                    tool_results_for_gemini.append({"function_response": {"name": tool_name, "response": tool_output_content}})
                
                gemini_history.append({"role": "user", "parts": tool_results_for_gemini}) 
                current_iteration += 1; continue 
            
            final_text_response = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')]) if candidate.content and candidate.content.parts else ""
            response_to_frontend = {"parts": [{"type": "chat", "role": "model", "parts": [{"type": "text", "text": final_text_response}]}]}
            return JSONResponse(content=response_to_frontend)
        
        # Max iterations reached
        timeout_response = {"parts": [{"type": "chat", "role": "model", "parts": [{"type": "text", "text": "Max tool iterations reached."}]}]}
        return JSONResponse(content=timeout_response)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        import traceback
        print(f"[/gemini/chat] Generic Error: {e}"); traceback.print_exc()
        # Return a generic error response in the expected chat format
        error_chat_part = {"type": "chat", "role": "model", "parts": [{"type": "text", "text": f"An error occurred: {str(e)}"}]}
        return JSONResponse(content={"parts": [error_chat_part]}, status_code=500)

@app.get("/")
async def root_mcp_main(): 
    return {"message": "ProductPod Backend Proxy (with FastMCP tools) is running."}

# Ensure startup event is managed correctly
if hasattr(app, 'router') and hasattr(app.router, 'on_startup'):
    # Clean up any old or potentially duplicated startup events by name
    app.router.on_startup = [e for e in app.router.on_startup if getattr(e, '__name__', '') not in ('startup_event_mcp_old', 'startup_event', 'startup_event_mcp')]
    app.router.on_startup.append(startup_event_mcp) # Add the current one
elif hasattr(app, 'router') and not hasattr(app.router, 'on_startup'): # router exists but no on_startup list
     app.router.on_startup = [startup_event_mcp]
# If app.router doesn't exist, this simple script structure might not support on_startup easily without more context.

print("Backend main.py script (refactored with github_tools.py) defined.")
