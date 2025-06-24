# app/backend/github_tools.py

import base64
import random
import time
import traceback # For detailed error logging
import difflib # For generating diffs

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from github import Github, GithubException

# Assuming Context is defined elsewhere and will be imported in main.py
# from mcp.server.fastmcp import Context # This would cause circular if FastMCP also in main
# For now, use a forward reference or type hint string if Context is complex
# For simplicity, let's assume Context can be imported or is a simple type for these tools.
# If Context is from FastMCP in main.py, this file will need to be structured carefully
# or Context needs to be passed as 'Any' or a forward reference string 'Context'.
# Let's proceed with 'Any' for now if direct import is an issue.
from mcp.server.fastmcp import Context # This should be fine if mcp_app is in main.py

# --- User-Specific GitHub Client Initialization ---
def get_github_client_for_user(user_github_token: Optional[str]) -> Optional[Github]:
    """Initializes a PyGithub client with the provided user token."""
    if not user_github_token:
        print("Error: No GitHub token provided for user client initialization.")
        return None
    try:
        client = Github(user_github_token)
        return client
    except Exception as e:
        print(f"Error initializing PyGithub client with user token: {e}")
        return None

def get_repo_client(repo_name_full: str, user_github_token: Optional[str]) -> Optional[Any]: # Returns github.Repository.Repository
    """Gets a GitHub repository object using a user-specific GitHub token."""
    g_user = get_github_client_for_user(user_github_token)
    if not g_user:
        return None
    try:
        return g_user.get_repo(repo_name_full)
    except GithubException as e:
        print(f"Error getting repo '{repo_name_full}' using user token: {e.status} - {e.data.get('message', str(e))}")
        return None
    except Exception as e:
        print(f"Unexpected error getting repo '{repo_name_full}' with user token: {str(e)}")
        return None

# --- Helper to retrieve user token from context ---
def _get_user_token_from_context(context: Context, tool_name: str) -> Optional[str]:
    user_token = None
    product_pod_user_id_for_log = "unknown_user"
    if hasattr(context, 'request_context'):
        if isinstance(context.request_context, dict):
            user_token = context.request_context.get("user_github_token")
            product_pod_user_id_for_log = context.request_context.get("product_pod_user_id", product_pod_user_id_for_log)
        elif hasattr(context.request_context, 'user_github_token'):
             user_token = context.request_context.user_github_token
             if hasattr(context.request_context, 'product_pod_user_id'):
                 product_pod_user_id_for_log = context.request_context.product_pod_user_id
    elif hasattr(context, 'user_github_token'):
        user_token = context.user_github_token
        if hasattr(context, 'product_pod_user_id'):
            product_pod_user_id_for_log = context.product_pod_user_id
    if not user_token:
        context.error(f"[{tool_name}] Critical: User GitHub token not found in context for user '{product_pod_user_id_for_log}'. Authorization is required.")
    return user_token

# --- Conceptual Logging for Sandbox Creation ---
def log_sandbox_creation(
    context: Context,
    product_pod_user_id: str,
    sandbox_repo_fullname: str,
    sandbox_type: str,
    upstream_repo_fullname: Optional[str] = None
):
    log_message = (f"User: {product_pod_user_id}, SandboxRepo: {sandbox_repo_fullname}, Type: {sandbox_type}")
    if upstream_repo_fullname: log_message += f", Upstream: {upstream_repo_fullname}"
    context.info(f"[SandboxTracking] Record creation: {log_message}")

# --- Tool Definitions ---

# fetchFiles Tool
class FetchFilesInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name, including owner (e.g., 'owner/repo')")
    branch: Optional[str] = Field(default="main", description="Git branch to fetch from (default is 'main')")
    paths: List[str] = Field(description="List of full file paths to fetch from the repository root.")
class FetchFilesOutput(BaseModel):
    files: Dict[str, str] = Field(description="Mapping of file paths to content")
    error: Optional[str] = Field(default=None, description="Error message if fetching failed")

def fetch_files_mcp_tool(input: FetchFilesInput, context: Context) -> FetchFilesOutput:
    tool_name = "fetchFiles"
    context.info(f"[{tool_name}] Called for repo: {input.repo_name}, branch: {input.branch}, paths: {input.paths}")
    user_token = _get_user_token_from_context(context, tool_name)
    if not user_token: return FetchFilesOutput(files={}, error="User GitHub token not available; authorization required.")
    repo = get_repo_client(input.repo_name, user_github_token=user_token)
    if not repo: return FetchFilesOutput(files={}, error=f"Failed to get GitHub repository '{input.repo_name}'.")
    results: Dict[str, str] = {}; errors: list[str] = []
    for path_item in input.paths:
        try:
            file_content_or_list = repo.get_contents(path_item, ref=input.branch)
            if isinstance(file_content_or_list, list):
                err_detail = f"Path '{path_item}' is a directory."
                errors.append(err_detail); results[path_item] = f"ERROR: {err_detail}"; continue
            file_data = file_content_or_list
            if hasattr(file_data, 'type') and file_data.type != 'file':
                err_detail = f"Path '{path_item}' is not a file (type: {file_data.type})."
                errors.append(err_detail); results[path_item] = f"ERROR: {err_detail}"; continue
            if isinstance(file_data.content, str): content = file_data.content
            elif file_data.encoding == "base64" and file_data.content:
                try: content = base64.b64decode(file_data.content).decode("utf-8")
                except UnicodeDecodeError:
                    err_detail = f"Error decoding file '{path_item}': Not valid UTF-8."
                    errors.append(err_detail); results[path_item] = f"ERROR: {err_detail}"; continue
            elif not file_data.content and file_data.encoding is None and file_data.size == 0: content = ""
            else:
                err_detail = f"File '{path_item}' has no decodable content."
                errors.append(err_detail); results[path_item] = f"ERROR: {err_detail}"; continue
            results[path_item] = content
        except GithubException as e:
            err_detail = f"Error fetching file '{path_item}': {e.status} - {e.data.get('message', str(e))}"
            if e.status == 404: err_detail = f"File or path '{path_item}' not found."
            errors.append(err_detail); results[path_item] = f"ERROR: {err_detail}"
        except Exception as e:
            err_detail = f"Unexpected error fetching file '{path_item}': {str(e)}"
            errors.append(err_detail); results[path_item] = f"ERROR: {err_detail}"
    final_error_message = None
    if errors:
        if not any(k for k,v in results.items() if not v.startswith("ERROR:")): final_error_message = "All files failed: " + "; ".join(errors)
        else: final_error_message = "Some files failed: " + "; ".join(errors)
    return FetchFilesOutput(files=results, error=final_error_message)

# listFiles Tool
class ListFilesInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name (owner/repo).")
    branch: Optional[str] = Field(default="main", description="Git branch.")
    path: str = Field(default="", description="Directory path (defaults to root).")
class ListFilesOutput(BaseModel):
    items: List[Dict[str, str]] = Field(description="List of items ('name', 'type', 'path').")
    error: Optional[str] = Field(default=None)

def list_files_mcp_tool(input: ListFilesInput, context: Context) -> ListFilesOutput:
    tool_name = "listFiles"; context.info(f"[{tool_name}] Repo: {input.repo_name}, Path: '{input.path}'")
    user_token = _get_user_token_from_context(context, tool_name)
    if not user_token: return ListFilesOutput(items=[], error="User GitHub token not available.")
    repo = get_repo_client(input.repo_name, user_github_token=user_token)
    if not repo: return ListFilesOutput(items=[], error=f"Failed to get repo '{input.repo_name}'.")
    try:
        contents = repo.get_contents(input.path, ref=input.branch)
        items = []
        if isinstance(contents, list):
            for item in contents: items.append({"name": item.name, "type": item.type, "path": item.path})
        elif contents: # Single item (e.g. file at root if path was "")
            items.append({"name": contents.name,"type": contents.type,"path": contents.path})
        return ListFilesOutput(items=items)
    except GithubException as e:
        err = f"Error listing path '{input.path}': {e.status} - {e.data.get('message', str(e))}"
        if e.status == 404: err = f"Path '{input.path}' not found."
        return ListFilesOutput(items=[], error=err)
    except Exception as e: return ListFilesOutput(items=[], error=f"Unexpected error: {str(e)}")

# searchFilesInRepo Tool
class SearchFilesInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name (owner/repo).")
    query: str = Field(description="Search query (keywords, qualifiers).")
class SearchFilesOutput(BaseModel):
    results: List[Dict[str, str]] = Field(description="Found files ('name', 'path', 'sha', 'url').")
    total_count: int = Field(description="Total results found by API.")
    error: Optional[str] = Field(default=None)

def search_files_in_repo_mcp_tool(input: SearchFilesInput, context: Context) -> SearchFilesOutput:
    tool_name = "searchFilesInRepo"; context.info(f"[{tool_name}] Repo: {input.repo_name}, Query: '{input.query}'")
    user_token = _get_user_token_from_context(context, tool_name)
    if not user_token: return SearchFilesOutput(results=[],total_count=0,error="User GitHub token not available.")
    g_user = get_github_client_for_user(user_token)
    if not g_user: return SearchFilesOutput(results=[],total_count=0,error="Failed to init GitHub client.")
    try:
        search_results = g_user.search_code(query=f"repo:{input.repo_name} {input.query}")
        items = [{"name": cf.name,"path": cf.path,"sha": cf.sha,"url": cf.html_url} for i, cf in enumerate(search_results) if i < 30]
        return SearchFilesOutput(results=items, total_count=search_results.totalCount)
    except GithubException as e: return SearchFilesOutput(results=[],total_count=0,error=f"GitHub API error: {e.status} - {e.data.get('message', str(e))}")
    except Exception as e: return SearchFilesOutput(results=[],total_count=0,error=f"Unexpected error: {str(e)}")

# gitCommit Tool (Multi-file)
class FileCommitData(BaseModel):
    path: str = Field(description="Full path to the file.")
    content: str = Field(description="New UTF-8 content.")
class MultiFileGitCommitInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name (owner/repo).")
    branch: str = Field(description="Git branch to commit to.")
    message: str = Field(description="Commit message.")
    files: List[FileCommitData] = Field(description="Files to create/update.")
class GitCommitOutput(BaseModel):
    commit_url: Optional[str] = Field(default=None)
    commit_sha: Optional[str] = Field(default=None)
    status: str
    error: Optional[str] = Field(default=None)

def git_commit_mcp_tool(input: MultiFileGitCommitInput, context: Context) -> GitCommitOutput:
    tool_name = "gitCommit"; context.info(f"[{tool_name}] Repo: {input.repo_name}, Branch: {input.branch}, Files: {len(input.files)}")
    user_token = _get_user_token_from_context(context, tool_name)
    if not user_token: return GitCommitOutput(status="failure", error="User GitHub token not available.")
    repo = get_repo_client(input.repo_name, user_github_token=user_token)
    if not repo: return GitCommitOutput(status="failure", error=f"Failed to get repo '{input.repo_name}'.")
    if not input.files: return GitCommitOutput(status="failure", error="No files for commit.")
    try:
        branch_ref = repo.get_git_ref(f"heads/{input.branch}")
        parent_commit_sha = branch_ref.object.sha
        parent_commit = repo.get_git_commit(parent_commit_sha)
        base_tree_sha = parent_commit.tree.sha
    except GithubException as e: # Branch not found or other issue
        if e.status == 404 and input.branch == repo.default_branch: # Committing to empty default branch
            parent_commit_sha, base_tree_sha = None, None
        else: return GitCommitOutput(status="failure", error=f"Branch '{input.branch}' not found or error: {str(e)}")
    tree_elements = []
    for file_data in input.files:
        blob = repo.create_git_blob(content=file_data.content, encoding="utf-8")
        tree_elements.append({"path": file_data.path, "mode": "100644", "type": "blob", "sha": blob.sha})
    new_tree_obj = repo.create_git_tree(tree_elements, base_tree=base_tree_sha) if base_tree_sha else repo.create_git_tree(tree_elements)
    parents = [repo.get_git_commit(parent_commit_sha)] if parent_commit_sha else []
    created_commit = repo.create_git_commit(message=input.message, tree=new_tree_obj, parents=parents)
    if parent_commit_sha: branch_ref.edit(sha=created_commit.sha)
    else: repo.create_git_ref(ref=f"refs/heads/{input.branch}", sha=created_commit.sha)
    return GitCommitOutput(commit_url=created_commit.html_url, commit_sha=created_commit.sha, status="success")
    except GithubException as e: return GitCommitOutput(status="failure", error=f"GitHub API error: {e.status} - {e.data.get('message', str(e))}")
    except Exception as e: traceback.print_exc(); return GitCommitOutput(status="failure", error=f"Unexpected error: {str(e)}")

# previewChanges Tool
class PreviewChangesInput(BaseModel):
    repo_name: str = Field(description="GitHub repository name (owner/repo).")
    branch: str = Field(description="Git branch for proposed changes.")
    files: List[FileCommitData] = Field(description="File changes to preview.")
class PreviewChangesOutput(BaseModel):
    diff_text: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)

def preview_changes_mcp_tool(input: PreviewChangesInput, context: Context) -> PreviewChangesOutput:
    tool_name = "previewChanges"; context.info(f"[{tool_name}] Repo: {input.repo_name}, Branch: {input.branch}, Files: {len(input.files)}")
    user_token = _get_user_token_from_context(context, tool_name)
    if not user_token: return PreviewChangesOutput(error="User GitHub token not available.")
    repo = get_repo_client(input.repo_name, user_github_token=user_token)
    if not repo: return PreviewChangesOutput(error=f"Failed to get repo '{input.repo_name}'.")
    if not input.files: return PreviewChangesOutput(diff_text="No file changes for preview.")
    full_diff_text = []
    for file_data in input.files:
        original_content_lines = []; original_file_exists = True
        try:
            existing_file = repo.get_contents(file_data.path, ref=input.branch)
            if isinstance(existing_file, list): # Path is a directory
                 full_diff_text.extend([f"--- a/{file_data.path}\n",f"+++ b/{file_data.path}\n",f"@@ -0,0 +1,1 @@\n",f"+Error: Path '{file_data.path}' is a directory.\n"]); continue
            if hasattr(existing_file,'type') and existing_file.type != 'file':
                 full_diff_text.extend([f"--- a/{file_data.path}\n",f"+++ b/{file_data.path}\n",f"@@ -0,0 +1,1 @@\n",f"+Error: Path '{file_data.path}' not a file (type: {existing_file.type}).\n"]); continue
            original_content = base64.b64decode(existing_file.content).decode('utf-8',errors='replace') if existing_file.encoding=="base64" and existing_file.content else existing_file.content if isinstance(existing_file.content,str) else "" if existing_file.size == 0 else None
            if original_content is None: full_diff_text.extend([f"--- a/{file_data.path}\n",f"+++ b/{file_data.path}\n",f"@@ -0,0 +1,1 @@\n",f"+Binary or unreadable: {file_data.path}\n"]); continue
            original_content_lines = original_content.splitlines(keepends=True)
        except GithubException as e:
            if e.status == 404: original_file_exists = False; original_content_lines = []
            else: return PreviewChangesOutput(error=f"Error fetching original '{file_data.path}': {str(e)}")
        new_content_lines = file_data.content.splitlines(keepends=True)
        diff = difflib.unified_diff(original_content_lines,new_content_lines,fromfile="/dev/null" if not original_file_exists else f"a/{file_data.path}",tofile=f"b/{file_data.path}",lineterm='\n')
        full_diff_text.extend(list(diff))
    if not full_diff_text and any("Error:" in line for line in full_diff_text): pass
    elif not full_diff_text : return PreviewChangesOutput(diff_text="No textual changes or all files problematic.")
    return PreviewChangesOutput(diff_text="".join(full_diff_text))

# sandboxInit - Helper for project init (now simplified)
def run_project_initialization_logic(project_name: str, dest_repo_name_full: str, project_description: str, dest_branch: str, user_token: str, context: Context):
    tool_name = "sandboxInit.run_project_initialization_logic"
    g_user = get_github_client_for_user(user_token)
    if not g_user: raise Exception("GitHub client could not be initialized for project init.")
    project_repo = get_repo_client(dest_repo_name_full, user_github_token=user_token)
    if not project_repo: raise Exception(f"Destination repo '{dest_repo_name_full}' not accessible.")
    context.info(f"[{tool_name}] Project '{project_name}' setup on branch '{dest_branch}' in repo '{dest_repo_name_full}' is complete (repository starts empty).")

# sandboxInit Tool
class SandboxInitInput(BaseModel):
    upstream_repo_name: Optional[str] = Field(default=None, description="The 'owner/repo' to be forked.")
    new_sandbox_repo_name: Optional[str] = Field(default=None, description="Name for a new temporary repo.")
    new_repo_description: Optional[str] = Field(default="AI sandbox repository.")
    new_repo_private: Optional[bool] = Field(default=True)
    branch_name: Optional[str] = Field(default=None, description="Branch to create/use. Auto-generated if None.")
    initialize_project_structure: Optional[bool] = Field(default=False)
    project_name_for_init: Optional[str] = Field(default="default_ai_project")
    project_description_for_init: Optional[str] = Field(default="AI initialized project.")
    reuse_token: Optional[str] = Field(default=None, description="Token to reuse a sandbox.")
    force_new: Optional[bool] = Field(default=False)
    direct_target_repo_name: Optional[str] = Field(default=None, description="Operate directly on this 'owner/repo'.")
class SandboxInitOutput(BaseModel):
    repo_name: Optional[str] = Field(default=None)
    branch: Optional[str] = Field(default=None)
    reuse_token: Optional[str] = Field(default=None)
    message: str
    error: Optional[str] = Field(default=None)
    is_fork: Optional[bool] = Field(default=None)
    is_new_repo: Optional[bool] = Field(default=None)
    project_name_initialized: Optional[str] = Field(default=None)

def sandbox_init_mcp_tool(input: SandboxInitInput, context: Context) -> SandboxInitOutput:
    tool_name = "sandboxInit"; context.info(f"[{tool_name}] Input: {input.model_dump_json(exclude_none=True)}")
    user_token = _get_user_token_from_context(context, tool_name)
    if not user_token: return SandboxInitOutput(message="Init failed", error="User GitHub token not available.")
    g_user = get_github_client_for_user(user_token)
    if not g_user: return SandboxInitOutput(message="Init failed", error="Failed to init GitHub client.")
    try: authenticated_user = g_user.get_user(); auth_user_login = authenticated_user.login
    except GithubException as e: return SandboxInitOutput(message="Init failed", error=f"Invalid token or perms: {str(e)}")

    target_repo:Optional[Any]=None; final_branch_name:Optional[str]=input.branch_name
    is_fork_flag,is_new_repo_created_flag,is_new_fork_created_flag = False,False,False
    sandbox_type_msg, proj_init_msg = "", ""
    orig_repo_for_log:Optional[Any]=None

    prod_user_id="unknown_user" # Simplified product_pod_user_id retrieval
    if hasattr(context,'request_context') and isinstance(context.request_context,dict): prod_user_id=context.request_context.get("product_pod_user_id",prod_user_id)
    if prod_user_id=="unknown_user": context.warning(f"[{tool_name}] ProductPod User ID not found.")

    try:
        if input.reuse_token and not input.force_new:
            try:
                reused_repo_name, reused_branch_name = base64.urlsafe_b64decode(input.reuse_token.encode()).decode().split(":",1)
                target_repo = get_repo_client(reused_repo_name, user_token)
                if target_repo: target_repo.get_branch(reused_branch_name); final_branch_name=reused_branch_name
                else: context.warning(f"[{tool_name}] Failed to get repo from reuse_token.")
                if target_repo: sandbox_type_msg = f"Reusing {'fork' if target_repo.fork else 'repo'} '{target_repo.full_name}'."
            except Exception as e: context.warning(f"[{tool_name}] Invalid reuse_token: {e}.")

        if not target_repo: # Determine target_repo if not reusing
            if input.direct_target_repo_name:
                target_repo = get_repo_client(input.direct_target_repo_name, user_token)
                if not target_repo: return SandboxInitOutput(message="Init failed", error=f"Direct target repo not found.")
                if auth_user_login and target_repo.owner.login != auth_user_login: context.warning(f"User operating on non-owned direct repo.")
                sandbox_type_msg = f"Using direct target repo '{target_repo.full_name}'."
            elif input.upstream_repo_name:
                orig_repo_for_log = get_repo_client(input.upstream_repo_name, user_token)
                if not orig_repo_for_log: return SandboxInitOutput(message="Init failed", error=f"Upstream repo not found.")
                if auth_user_login and orig_repo_for_log.owner.login == auth_user_login:
                    target_repo = orig_repo_for_log; sandbox_type_msg = f"Using user's own repo '{target_repo.full_name}'."
                else: # Fork or use existing
                    expected_fork_name = f"{auth_user_login}/{orig_repo_for_log.name}"
                    try: fork_repo = g_user.get_repo(expected_fork_name)
                    except GithubException as e: if e.status != 404: raise; fork_repo = None
                    if fork_repo and fork_repo.fork and fork_repo.parent and fork_repo.parent.full_name == orig_repo_for_log.full_name: target_repo = fork_repo
                    else: # Create fork
                        orig_repo_for_log.create_fork(); is_new_fork_created_flag = True; time.sleep(10) # Allow time for fork
                        target_repo = get_repo_client(expected_fork_name, user_token) # Re-fetch
                        if not target_repo : return SandboxInitOutput(message="Init failed", error="Fork initiated but not found.")
                    is_fork_flag = True; sandbox_type_msg = f"Using fork '{target_repo.full_name}'."
                    if is_new_fork_created_flag: log_sandbox_creation(context,prod_user_id,target_repo.full_name,"fork",orig_repo_for_log.full_name)
            elif input.new_sandbox_repo_name:
                full_new_name = f"{auth_user_login}/{input.new_sandbox_repo_name}"
                try: target_repo = g_user.get_repo(full_new_name)
                except GithubException as e:
                    if e.status == 404:
                        target_repo = authenticated_user.create_repo(name=input.new_sandbox_repo_name,description=input.new_repo_description,private=input.new_repo_private,auto_init=True)
                        is_new_repo_created_flag = True; log_sandbox_creation(context,prod_user_id,target_repo.full_name,"temp_repo")
                    else: raise
                sandbox_type_msg = f"Using repo '{target_repo.full_name}'."
            else: return SandboxInitOutput(message="Init failed", error="Must specify repo source.")

        if not target_repo: return SandboxInitOutput(message="Init failed", error="Target repo not determined.")

        default_branch = target_repo.get_branch(target_repo.default_branch); base_sha = default_branch.commit.sha
        if not final_branch_name: # Auto-generate
            final_branch_name=f"sandbox-{random.choice(['sky','sea','sun'])}-{random.randint(100,999)}" # Simpler name
            target_repo.create_git_ref(ref=f"refs/heads/{final_branch_name}",sha=base_sha)
        else: # Ensure branch exists
            try: target_repo.get_branch(final_branch_name)
            except GithubException as e:
                if e.status==404: target_repo.create_git_ref(ref=f"refs/heads/{final_branch_name}",sha=base_sha)
                else: raise

        if input.initialize_project_structure:
            run_project_initialization_logic(input.project_name_for_init,target_repo.full_name,input.project_description_for_init,final_branch_name,user_token,context)
            proj_init_msg = f"Project '{input.project_name_for_init}' init done. "

        reuse_tok = base64.urlsafe_b64encode(f"{target_repo.full_name}:{final_branch_name}".encode()).decode()
        return SandboxInitOutput(branch=final_branch_name,repo_name=target_repo.full_name,reuse_token=reuse_tok,
            message=f"{sandbox_type_msg} Branch '{final_branch_name}' ready. {proj_init_msg}Reuse: {reuse_tok}",
            is_fork=is_fork_flag,is_new_repo=is_new_repo_created_flag,
            project_name_initialized=input.project_name_for_init if input.initialize_project_structure else None)
    except Exception as e:
        traceback.print_exc(); return SandboxInitOutput(message="Init failed", error=str(e))
