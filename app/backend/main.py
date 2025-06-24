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

# PyGithub and other necessary imports for direct GitHub operations
from github import Github, GithubException
import base64
import random # for sandboxInit

# Initialize GitHub client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("CRITICAL ERROR: GITHUB_TOKEN environment variable not set.")
    # Potentially raise an error or exit if GITHUB_TOKEN is essential for startup
    # For now, we'll let it proceed but tools will fail if GITHUB_TOKEN is None.
    g = None
else:
    g = Github(GITHUB_TOKEN)

# Helper to get a repo object - adapting from project/sample/main.py
def get_repo_client(repo_name_full: str) -> Optional[Any]: # Actually returns github.Repository.Repository
    if not g:
        print("Error: GitHub client not initialized (GITHUB_TOKEN missing).")
        return None
    try:
        # Assuming repo_name_full is in "owner/repo" format
        return g.get_repo(repo_name_full)
    except GithubException as e:
        print(f"Error getting repo {repo_name_full}: {e}")
        return None

class FetchFilesInput(BaseModel):
    # repo_name in "owner/repo" format as per Pydantic model,
    # but original sample code used "ai-delivery-framework" and then constructed owner/repo.
    # Will need to ensure consistency or adapt. For now, assume "owner/repo" is passed.
    repo_name: str = Field(description="GitHub repository name, including owner (e.g., 'owner/repo')")
    branch: Optional[str] = Field(default="main", description="Git branch to fetch from (default is 'main')")
    paths: List[str] = Field(description="List of full file paths to fetch from the repository root.")

class FetchFilesOutput(BaseModel): 
    files: Dict[str, str] = Field(description="Mapping of file paths to content")
    error: Optional[str] = Field(default=None, description="Error message if fetching failed")

@mcp_app.tool(name="fetchFiles")
def fetch_files_mcp_tool(input: FetchFilesInput, context: Context) -> FetchFilesOutput:
    """Fetches content of specified files from a GitHub repository and branch."""
    context.info(f"[fetch_files_mcp_tool] Called for repo: {input.repo_name}, branch: {input.branch}, paths: {input.paths}")

    repo = get_repo_client(input.repo_name)
    if not repo:
        err_msg = f"Failed to get GitHub repository: {input.repo_name}"
        context.error(f"[fetch_files_mcp_tool] {err_msg}")
        return FetchFilesOutput(files={}, error=err_msg)

    results: Dict[str, str] = {}
    errors: list[str] = []

    for path in input.paths:
        try:
            file_content = repo.get_contents(path, ref=input.branch)
            if isinstance(file_content.content, str):
                 # Already decoded (e.g. if it's not base64 encoded)
                content = file_content.content
            elif file_content.encoding == "base64" and file_content.content:
                content = base64.b64decode(file_content.content).decode("utf-8")
            else: # No content or unknown encoding
                content = "" # Or handle as error, TBD
                errors.append(f"File '{path}' has no content or unknown encoding.")
            results[path] = content
            context.info(f"[fetch_files_mcp_tool] Fetched file: {path}")
        except GithubException as e:
            err_detail = f"Error fetching file '{path}': {e.status} - {e.data.get('message', str(e))}"
            context.error(f"[fetch_files_mcp_tool] {err_detail}")
            errors.append(err_detail)
            results[path] = f"ERROR: {err_detail}" # Include error in content for partial success
        except Exception as e: # Catch other unexpected errors
            err_detail = f"Unexpected error fetching file '{path}': {str(e)}"
            context.error(f"[fetch_files_mcp_tool] {err_detail}")
            errors.append(err_detail)
            results[path] = f"ERROR: {err_detail}"


    if errors and not results: # Only errors, no files successfully fetched
         return FetchFilesOutput(files={}, error="; ".join(errors))
    elif errors: # Partial success
        # Return files dictionary which includes error messages for failed files,
        # and also an overall error message.
        return FetchFilesOutput(files=results, error="Some files could not be fetched: " + "; ".join(errors))

    return FetchFilesOutput(files=results)

class GitCommitInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name, including owner (e.g., 'owner/repo')")
    branch: str = Field(description="Git branch to commit to (e.g., 'main', 'develop')")
    message: str = Field(description="The commit message.")
    path: str = Field(description="The full path to the file in the repository to create or update.")
    content: str = Field(description="The new content of the file.")

class GitCommitOutput(BaseModel):
    commit_url: Optional[str] = Field(default=None, description="URL of the new commit if successful.")
    commit_sha: Optional[str] = Field(default=None, description="SHA of the new commit if successful.")
    status: str = Field(description="Status of the commit operation (e.g., 'success', 'failure').")
    error: Optional[str] = Field(default=None, description="Error message if the commit failed.")

@mcp_app.tool(name="gitCommit")
def git_commit_mcp_tool(input: GitCommitInput, context: Context) -> GitCommitOutput:
    """Creates or updates a file in a GitHub repository and commits the change."""
    context.info(f"[git_commit_mcp_tool] Called for repo: {input.repo_name}, branch: {input.branch}, path: {input.path}")

    repo = get_repo_client(input.repo_name)
    if not repo:
        err_msg = f"Failed to get GitHub repository: {input.repo_name}"
        context.error(f"[git_commit_mcp_tool] {err_msg}")
        return GitCommitOutput(status="failure", error=err_msg)

    try:
        # Check if file exists to decide between update or create
        try:
            existing_file = repo.get_contents(input.path, ref=input.branch)
            commit_details = repo.update_file(
                path=input.path,
                message=input.message,
                content=input.content,
                sha=existing_file.sha,
                branch=input.branch
            )
            context.info(f"[git_commit_mcp_tool] Updated file: {input.path}")
        except GithubException as e:
            if e.status == 404: # File not found, so create it
                commit_details = repo.create_file(
                    path=input.path,
                    message=input.message,
                    content=input.content,
                    branch=input.branch
                )
                context.info(f"[git_commit_mcp_tool] Created new file: {input.path}")
            else: # Other GitHub error during get_contents
                raise e

        commit_obj = commit_details['commit']

        return GitCommitOutput(
            commit_url=commit_obj.html_url,
            commit_sha=commit_obj.sha,
            status="success"
        )

    except GithubException as e:
        err_detail = f"GitHub API error: {e.status} - {e.data.get('message', str(e))}"
        context.error(f"[git_commit_mcp_tool] {err_detail}")
        return GitCommitOutput(status="failure", error=err_detail)
    except Exception as e:
        err_detail = f"Unexpected error during commit: {str(e)}"
        context.error(f"[git_commit_mcp_tool] {err_detail}")
        return GitCommitOutput(status="failure", error=err_detail)

class SandboxInitInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name, including owner (e.g., 'owner/repo')")
    mode: str = Field(description="Mode of operation: 'branch' or 'project'.")
    reuse_token: Optional[str] = Field(default=None); force_new: Optional[bool] = Field(default=False)
    branch: Optional[str] = Field(default=None); project_name: Optional[str] = Field(default=None)
    project_description: Optional[str] = Field(default=None)

class SandboxInitOutput(BaseModel):
    branch: Optional[str] = Field(default=None); repo_name: Optional[str] = Field(default=None)
    reuse_token: Optional[str] = Field(default=None); message: str
    error: Optional[str] = Field(default=None)

# --- Helper functions for sandboxInit (project mode) ---
# These are adapted from project/sample/main.py
# Note: These are synchronous and will run in FastMCP's thread pool if the tool is sync.

def copy_framework_baseline(source_repo, destination_repo, source_path, dest_path, destination_branch):
    """Recursively copy files and folders from the source repo to the destination repo."""
    # Assuming g (Github client) is globally available and configured
    contents = source_repo.get_contents(source_path) # Removed ref=source_repo.default_branch, assuming source_path is absolute or relative to root
    for item in contents:
        if item.type == "dir":
            new_dest_path = f"{dest_path}/{item.name}" if dest_path else item.name
            copy_framework_baseline(source_repo, destination_repo, item.path, new_dest_path, destination_branch)
        else:
            try:
                file_content_bytes = source_repo.get_contents(item.path).decoded_content
                file_content = file_content_bytes.decode('utf-8')
                # Adjusted destination_path to be relative to the root of the destination_repo
                # The original logic for framework had "framework/" prefix, we might not want that here
                # For now, let's assume dest_path is where it should go, or root if dest_path is empty
                final_destination_path = f"{dest_path}/{item.name}" if dest_path else item.name

                try:
                    existing_file = destination_repo.get_contents(final_destination_path, ref=destination_branch)
                    destination_repo.update_file(final_destination_path, f"Updated {item.name} from framework baseline", file_content, existing_file.sha, branch=destination_branch)
                except GithubException as e:
                    if e.status == 404: # Not found
                        destination_repo.create_file(final_destination_path, f"Copied {item.name} from framework baseline", file_content, branch=destination_branch)
                    else:
                        raise
            except UnicodeDecodeError:
                print(f"⚠️ Skipping binary file during copy: {item.path}")
            except Exception as e:
                print(f"⚠️ Error copying file {item.path}: {str(e)}")


def create_initial_project_files(project_repo, project_base_path, project_name, project_description, destination_branch):
    from datetime import datetime # Ensure datetime is imported
    starter_task_yaml = f"""tasks:
  1.1_capture_project_goals:
    description: Help capture and summarize the goals, purpose, and intended impact of the project.
    phase: Phase 1 - Discovery
    category: discovery
    pod_owner: DeliveryPod
    status: pending
    prompt: prompts/used/{project_name}_capture_project_goals_prompt.txt
    inputs: []
    outputs:
      - outputs/project_goals.md
    ready: true
    done: false
    created_by: human
    created_at: {datetime.utcnow().isoformat()}
    updated_at: {datetime.utcnow().isoformat()}
"""
    starter_memory_yaml = f"""memory:
  context:
    project_name: {project_name}
    project_description: {project_description}
    created_at: {datetime.utcnow().isoformat()}
"""
    files_to_create = {
        f"{project_base_path}/task.yaml": ("Initialize task.yaml", starter_task_yaml),
        f"{project_base_path}/memory.yaml": ("Initialize memory.yaml", starter_memory_yaml),
        f"{project_base_path}/outputs/.gitkeep": ("Initialize outputs folder", ""), # Ensure folder creation
        f"{project_base_path}/prompts/.gitkeep": ("Initialize prompts folder", ""),
        f"{project_base_path}/docs/.gitkeep": ("Initialize docs folder", ""),
    }
    for path, (message, content) in files_to_create.items():
        try:
            project_repo.create_file(path, message, content, branch=destination_branch)
        except GithubException as e:
            if e.status == 422 and "already exists" in e.data.get("message","").lower():
                print(f"File {path} already exists, skipping creation.")
            else:
                print(f"Error creating file {path}: {str(e)}")


def run_project_initialization_logic(project_name: str, dest_repo_name_full: str, project_description: str, dest_branch: str, context: Context):
    # This function encapsulates the logic from project/sample/main.py's run_project_initialization
    # It's made synchronous and uses the global 'g'
    if not g:
        context.error("GitHub client not initialized.")
        raise Exception("GitHub client not initialized (GITHUB_TOKEN missing).")

    try:
        # Assuming a fixed source "framework" repo for the baseline
        # This should be configurable or clearly defined. Using a placeholder for now.
        FRAMEWORK_OWNER_REPO = "stewmckendry/ai-delivery-framework" # Example, make this configurable if needed

        framework_repo = g.get_repo(FRAMEWORK_OWNER_REPO)
        project_repo = get_repo_client(dest_repo_name_full) # Use the existing helper

        if not project_repo:
            raise Exception(f"Destination repository {dest_repo_name_full} not found or accessible.")

        framework_source_path = "framework" # Path within the framework_repo to copy from
        project_base_dest_path = "project"  # Base path in the destination project repo, e.g. "project/"

        # 1. Validate framework source path exists (optional, get_contents will fail if not)
        try:
            framework_repo.get_contents(framework_source_path) # Check if path exists
        except GithubException as e:
            if e.status == 404:
                raise Exception(f"Framework source path '{framework_source_path}' not found in {FRAMEWORK_OWNER_REPO}.")
            raise

        # 2. Copy framework files from framework_repo's "framework/" path to project_repo's root (or a subfolder)
        # The original copy_framework_baseline copied to `framework/{dest_path}/{item.name}`.
        # If we want the contents of `framework_source_path` to go to the root of `project_repo`, `dest_path_for_copy` should be empty.
        # If they should go into e.g. "framework_files/" in project_repo, then "framework_files".
        # For now, let's assume files from FRAMEWORK_OWNER_REPO/framework/* go to project_repo/* (root)
        context.info(f"Copying framework baseline from {FRAMEWORK_OWNER_REPO}/{framework_source_path} to {dest_repo_name_full}@{dest_branch} root.")
        copy_framework_baseline(framework_repo, project_repo, source_path=framework_source_path, dest_path="", destination_branch=dest_branch)

        # 3. Create initial project-specific files (task.yaml, memory.yaml) under project_base_dest_path
        context.info(f"Creating initial project files in {project_base_dest_path}/ for {project_name}.")
        create_initial_project_files(project_repo, project_base_dest_path, project_name, project_description, destination_branch=dest_branch)

        context.info(f"Project {project_name} initialized successfully in {dest_repo_name_full} on branch {dest_branch}.")

    except GithubException as e:
        err_msg = f"GitHub error during project initialization: {e.status} - {e.data.get('message', str(e))}"
        context.error(err_msg)
        raise Exception(err_msg) # Re-raise to be caught by the tool's error handler
    except Exception as e:
        err_msg = f"Unexpected error during project initialization: {str(e)}"
        context.error(err_msg)
        import traceback
        traceback.print_exc()
        raise Exception(err_msg)


@mcp_app.tool(name="sandboxInit")
def sandbox_init_mcp_tool(input: SandboxInitInput, context: Context) -> SandboxInitOutput:
    """Initializes a sandbox environment: a new branch or a new project structure."""
    context.info(f"[sandbox_init_mcp_tool] Called for repo: {input.repo_name}, mode: {input.mode}")

    repo = get_repo_client(input.repo_name)
    if not repo:
        err_msg = f"Failed to get GitHub repository: {input.repo_name}"
        context.error(f"[sandbox_init_mcp_tool] {err_msg}")
        return SandboxInitOutput(message="Initialization failed", error=err_msg)

    if input.mode == "branch":
        try:
            base_branch_name = "main" # Or repo.default_branch

            # Try to get the default branch to ensure repo is valid and get its SHA
            try:
                default_branch_obj = repo.get_branch(base_branch_name)
                base_sha = default_branch_obj.commit.sha
            except GithubException as e:
                 # Fallback if 'main' doesn't exist, try default_branch
                try:
                    default_branch_obj = repo.get_branch(repo.default_branch)
                    base_branch_name = repo.default_branch
                    base_sha = default_branch_obj.commit.sha
                except GithubException as e_default:
                    err_msg = f"Cannot find base branch ('main' or default) in {input.repo_name}: {e_default.data.get('message', str(e_default))}"
                    context.error(f"[sandbox_init_mcp_tool] {err_msg}")
                    return SandboxInitOutput(message="Initialization failed", error=err_msg)


            new_branch_name = None
            if input.reuse_token and not input.force_new:
                try:
                    decoded_token = base64.urlsafe_b64decode(input.reuse_token.encode()).decode()
                    if decoded_token.startswith("sandbox-"):
                        repo.get_branch(decoded_token) # Check if branch exists
                        new_branch_name = decoded_token
                        context.info(f"Reusing existing sandbox branch: {new_branch_name}")
                except Exception: # Invalid token or branch doesn't exist
                    context.info("Invalid or non-existent reuse_token, will create a new branch.")
                    pass

            if not new_branch_name: # Create new branch
                # Simple unique name generation, similar to project/sample
                adj = ["emerald", "cosmic", "velvet", "silent", "curious", "ruby", "azure", "golden"]
                animals = ["hawk", "otter", "wave", "eagle", "fox", "lynx", "comet", "river"]

                for _ in range(5): # Try a few times for a unique name
                    candidate_branch = f"sandbox-{random.choice(adj)}-{random.choice(animals)}-{random.randint(100,999)}"
                    try:
                        repo.get_branch(candidate_branch) # Check if it exists
                    except GithubException as e: # If it doesn't exist (404), then it's usable
                        if e.status == 404:
                            new_branch_name = candidate_branch
                            break
                        else: raise # Other error

                if not new_branch_name: # Still couldn't find a unique name
                    new_branch_name = f"sandbox-{base64.urlsafe_b64encode(os.urandom(6)).decode().rstrip('=')}" # Fallback

                repo.create_git_ref(ref=f"refs/heads/{new_branch_name}", sha=base_sha)
                context.info(f"Created new sandbox branch: {new_branch_name} from {base_branch_name} ({base_sha})")

            current_reuse_token = base64.urlsafe_b64encode(new_branch_name.encode()).decode()
            return SandboxInitOutput(
                branch=new_branch_name,
                repo_name=input.repo_name,
                reuse_token=current_reuse_token,
                message=f"Sandbox branch '{new_branch_name}' is ready in repo '{input.repo_name}'. Reuse token: {current_reuse_token}"
            )
        except GithubException as e:
            err_detail = f"GitHub error during branch initialization: {e.status} - {e.data.get('message', str(e))}"
            context.error(f"[sandbox_init_mcp_tool] {err_detail}")
            return SandboxInitOutput(message="Branch initialization failed", error=err_detail)
        except Exception as e:
            err_detail = f"Unexpected error during branch initialization: {str(e)}"
            context.error(f"[sandbox_init_mcp_tool] {err_detail}")
            return SandboxInitOutput(message="Branch initialization failed", error=err_detail)

    elif input.mode == "project":
        if not input.branch or not input.project_name: # Description is optional
            err_msg = "For project mode, 'branch' and 'project_name' are required."
            context.error(f"[sandbox_init_mcp_tool] {err_msg}")
            return SandboxInitOutput(message="Initialization failed", error=err_msg)
        try:
            # Ensure the target branch for the project exists, or create it if not.
            # The project init logic expects the branch to exist.
            try:
                repo.get_branch(input.branch)
                context.info(f"Target branch '{input.branch}' for project init already exists.")
            except GithubException as e:
                if e.status == 404: # Branch not found, create it from default branch
                    try:
                        default_branch_sha = repo.get_branch(repo.default_branch).commit.sha
                        repo.create_git_ref(ref=f"refs/heads/{input.branch}", sha=default_branch_sha)
                        context.info(f"Created target branch '{input.branch}' for project initialization.")
                    except GithubException as ce:
                        err_msg = f"Failed to create target branch '{input.branch}': {ce.data.get('message', str(ce))}"
                        context.error(f"[sandbox_init_mcp_tool] {err_msg}")
                        return SandboxInitOutput(message="Initialization failed", error=err_msg)
                else: # Other error checking branch
                    raise

            # Run the synchronous project initialization logic
            run_project_initialization_logic(
                project_name=input.project_name,
                dest_repo_name_full=input.repo_name,
                project_description=input.project_description or f"Project {input.project_name}",
                dest_branch=input.branch,
                context=context
            )
            # reuse_token is not typically part of project init, more for branch persistence
            return SandboxInitOutput(
                branch=input.branch,
                repo_name=input.repo_name,
                project_name=input.project_name,
                message=f"Project '{input.project_name}' initialized in repo '{input.repo_name}' on branch '{input.branch}'."
            )
        except Exception as e: # Catch errors from run_project_initialization_logic or branch creation
            err_detail = f"Project initialization failed: {str(e)}"
            context.error(f"[sandbox_init_mcp_tool] {err_detail}")
            return SandboxInitOutput(message="Project initialization failed", error=err_detail)
    else:
        err_msg = f"Unsupported mode for sandboxInit: {input.mode}"
        context.error(f"[sandbox_init_mcp_tool] {err_msg}")
        return SandboxInitOutput(message="Initialization failed", error=err_msg)


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

async def get_gemini_tools_from_fastmcp(mcp_server: FastMCP) -> List[GeminiSdkTool]:
    gemini_function_declarations = []
    try:
        mcp_tools = await mcp_server.list_tools()
        if not mcp_tools:
            print("[get_gemini_tools_from_fastmcp] No MCP tools found.")
            return []

        for mcp_tool in mcp_tools:
            tool_name = mcp_tool.name
            tool_description = mcp_tool.description or f"Tool: {tool_name}"
            
            # Directly use the inputSchema generated by Pydantic via FastMCP
            # It should already be in the correct format (OpenAPI 3.0 JSON Schema)
            parameters_schema = copy.deepcopy(mcp_tool.inputSchema)

            # Ensure the schema is a dictionary and has a 'type' of 'object'.
            # Pydantic models for tool inputs should naturally produce this.
            if not isinstance(parameters_schema, dict) or parameters_schema.get("type") != "object":
                print(f"[get_gemini_tools_from_fastmcp] Warning: Schema for tool '{tool_name}' is not a direct object schema. Attempting to wrap.")
                # This case should ideally not happen if Pydantic models are used correctly for input.
                # If it's a simple type or something else, Gemini might not accept it directly.
                # Forcing it into an object structure might be incorrect.
                # However, Gemini expects a JSON schema object for parameters.
                # If inputSchema is None or not an object, we might need a default.
                parameters_schema = {"type": "object", "properties": {}} # Minimal valid schema

            # Remove 'title' from the top level of the schema if present, as Gemini uses tool name/description
            if 'title' in parameters_schema:
                del parameters_schema['title']

            print(f"--- [get_gemini_tools_from_fastmcp] Preparing tool for Gemini: {tool_name} ---")
            print(f"Description: {tool_description}")
            print(f"Parameters (from mcp_tool.inputSchema): {json.dumps(parameters_schema, indent=2)}")

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
    else:
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
