import logging
import os
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from fastmcp import FastMCP
from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import (
    NbModelClient,
    get_jupyter_notebook_websocket_url,
)

# Initialize FastMCP with a descriptive name and version
mcp = FastMCP(
    name="jupyter-notebook-executor",
    description="Execute and manage Jupyter notebooks with git sync capabilities and comprehensive EDA support",
    version="2.0.0",
    instructions="""
This server provides comprehensive Jupyter notebook execution and EDA capabilities with multi-server support.

Key Features:
- Execute code in persistent Jupyter notebooks  
- Complete EDA workflow support with templates and guidance
- Git integration for version control
- Variable inspection and kernel management
- AI-powered analysis recommendations
- Multi-server support with automatic token selection

Multi-Server Setup:
Set environment variables for your Jupyter servers:
- LOCAL_JUPYTER_TOKEN: For local Jupyter servers (localhost on any port)
- REMOTE_JUPYTER_TOKEN: For any remote Jupyter servers (non-localhost domains)

Token Selection Rules:
1. hostname == 'localhost' (any port) → LOCAL_JUPYTER_TOKEN
2. hostname != 'localhost' (any remote domain) → REMOTE_JUPYTER_TOKEN

Quick Start for EDA:
1. Set appropriate environment variables for your Jupyter servers
2. Provide your Jupyter connection info (notebook URL, kernel ID, etc.)
3. The server will automatically select the correct token based on hostname/port
4. Use EDA prompts for structured workflow guidance
5. Use templates for common analysis patterns

For help, access 'docs://eda-guide' resource.
""",
    on_duplicate_tools="warn",
    on_duplicate_resources="warn",
    on_duplicate_prompts="warn",
)

# Environment Configuration - Multiple Jupyter Server Support
LOCAL_JUPYTER_TOKEN = os.getenv("LOCAL_JUPYTER_TOKEN")
REMOTE_JUPYTER_TOKEN = os.getenv("REMOTE_JUPYTER_TOKEN")

logger = logging.getLogger(__name__)

# Check if any tokens are missing and warn
missing_tokens = []
if not LOCAL_JUPYTER_TOKEN:
    missing_tokens.append("LOCAL_JUPYTER_TOKEN")
if not REMOTE_JUPYTER_TOKEN:
    missing_tokens.append("REMOTE_JUPYTER_TOKEN")

if missing_tokens:
    logger.warning(
        f"Missing environment variables: {', '.join(missing_tokens)}. Some operations may fail."
    )


def extract_connection_info(connection_info: dict) -> tuple[str, str]:
    """
    Extract server URL and notebook path from connection info.

    Args:
        connection_info: Dict containing jupyter connection details

    Returns:
        tuple of (server_url, notebook_path)
    """
    notebook_path = connection_info.get("notebookPath", "")

    # Extract server URL from notebook path
    # Format: http://localhost:8890/project/notebooks/file.ipynb
    if notebook_path.startswith("http"):
        parts = notebook_path.split("/")
        server_url = f"{parts[0]}//{parts[2]}"  # http://localhost:8890
        # Extract path after server URL
        path_parts = "/".join(parts[3:])  # project/notebooks/file.ipynb
    else:
        server_url = "http://localhost:8888"  # fallback
        path_parts = notebook_path

    return server_url, path_parts


def get_token_for_server(server_url: str) -> str:
    """
    Dynamically choose the appropriate token based on server URL.

    Args:
        server_url: The Jupyter server URL (e.g., "http://localhost:8801")

    Returns:
        str: The appropriate token for the server

    Raises:
        ValueError: If no appropriate token is found or configured
    """
    parsed_url = urlparse(server_url)
    hostname = parsed_url.hostname

    # Rule 1: localhost -> LOCAL_JUPYTER_TOKEN
    if hostname == "localhost":
        if not LOCAL_JUPYTER_TOKEN:
            raise ValueError(f"LOCAL_JUPYTER_TOKEN not configured for server {server_url}")
        return LOCAL_JUPYTER_TOKEN

    # Rule 2: Any other hostname -> REMOTE_JUPYTER_TOKEN
    else:
        if not REMOTE_JUPYTER_TOKEN:
            raise ValueError(f"REMOTE_JUPYTER_TOKEN not configured for server {server_url}")
        return REMOTE_JUPYTER_TOKEN


def extract_output(output: dict) -> str:
    """
    Extracts readable output from a Jupyter cell output dictionary.

    Args:
        output (dict): The output dictionary from a Jupyter cell.

    Returns:
        str: A string representation of the output.
    """
    output_type = output.get("output_type")
    if output_type == "stream":
        return output.get("text", "")
    elif output_type in ["display_data", "execute_result"]:
        data = output.get("data", {})
        if "text/plain" in data:
            return data["text/plain"]
        elif "text/html" in data:
            return "[HTML Output]"
        elif "image/png" in data:
            return "[Image Output (PNG)]"
        else:
            return f"[{output_type} Data: keys={list(data.keys())}]"
    elif output_type == "error":
        return "\n".join(output.get("traceback", []))
    else:
        return f"[Unknown output type: {output_type}]"


@mcp.tool(
    description="Execute shell commands or Python code directly on remote server (NO notebook persistence)"
)
async def execute_remote_code(
    connection_info: dict,
    code: str,
    execution_type: str = "shell",
) -> dict[str, Union[bool, str, list[str]]]:
    """Execute shell commands or Python code directly on the remote server WITHOUT notebook persistence.

    This tool executes code directly through the kernel without creating or modifying notebook cells.
    The execution results are NOT saved to any notebook file. Use this for:

    WHEN TO USE:
    - Quick system operations and file management
    - Environment setup and package installation
    - One-off data inspection or debugging
    - Git operations and system administration
    - Testing code before adding to notebook
    - Shell commands that don't need to be saved

    WHEN NOT TO USE:
    - When you want the code/results saved in a notebook
    - For analysis steps that should be reproducible
    - When building a permanent workflow in a notebook

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)
        code: The shell command or Python code to execute
        execution_type: Either "shell" for shell commands or "python" for Python code

    Returns:
        dict containing execution results and outputs (NOT saved to notebook)
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client with provided info
        try:
            token = get_token_for_server(server_url)
        except Exception as token_error:
            return {
                "success": False,
                "error": f"Token selection failed: {token_error}",
                "execution_type": execution_type,
            }

        remote_kernel = KernelClient(server_url=server_url, token=token, kernel_id=kernel_id)
        remote_kernel.start()

        if execution_type == "shell":
            # Execute shell command with ! prefix
            result = remote_kernel.execute(f"!{code}")
        elif execution_type == "python":
            # Execute Python code directly
            result = remote_kernel.execute(code)
        else:
            return {
                "success": False,
                "error": f"Invalid execution_type: {execution_type}. Use 'shell' or 'python'",
            }

        # Extract the outputs from the result dictionary
        outputs = result.get("outputs", [])
        status = result.get("status", "unknown")
        execution_count = result.get("execution_count")

        # Handle different output formats
        if isinstance(outputs, list):
            # Extract output strings from the outputs list
            str_outputs = []
            for output in outputs:
                if isinstance(output, dict):
                    str_outputs.append(extract_output(output))
                else:
                    str_outputs.append(str(output))

            # Check for errors based on status or output types
            has_error = (status == "error") or any(
                isinstance(output, dict) and output.get("output_type") == "error"
                for output in outputs
            )
        else:
            # Handle unexpected output format
            str_outputs = [str(outputs)]
            has_error = status == "error"

        if has_error:
            # Find the first error output (only if it's a dict)
            error_output = None
            for output in outputs:
                if isinstance(output, dict) and output.get("output_type") == "error":
                    error_output = output
                    break

            if error_output:
                error_info = {
                    "error_name": error_output.get("ename", "Unknown Error"),
                    "error_value": error_output.get("evalue", ""),
                    "traceback": error_output.get("traceback", []),
                }
                return {
                    "success": False,
                    "outputs": str_outputs,
                    "execution_type": execution_type,
                    "error_info": error_info,
                    "status": status,
                    "execution_count": execution_count,
                }
            else:
                # Generic error handling if no proper error dict found
                return {
                    "success": False,
                    "outputs": str_outputs,
                    "execution_type": execution_type,
                    "error_info": {
                        "error_name": "Unknown",
                        "error_value": f"Execution status: {status}",
                    },
                    "status": status,
                    "execution_count": execution_count,
                }

        return {
            "success": True,
            "outputs": str_outputs,
            "execution_type": execution_type,
            "status": status,
            "execution_count": execution_count,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "execution_type": execution_type}


@mcp.tool(description="Pull latest changes from remote to the remote workspace")
async def pull_local_changes_on_remote(
    connection_info: dict,
) -> dict[str, Union[bool, str]]:
    """Pull latest changes from remote to the remote workspace.

    Args:
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing pull status
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        remote_kernel.start()
        # Get current working directory using pwd
        pwd_output = remote_kernel.execute("!pwd")
        current_dir = ""
        if pwd_output and pwd_output.get("outputs"):
            current_dir = extract_output(pwd_output["outputs"][0]).strip()
        
        if not current_dir:
            return {"success": False, "message": "Could not determine current working directory"}

        # Determine project directory by going up from notebooks directory
        if "notebooks" in current_dir:
            # If we're in a notebooks directory, go up to find project root
            remote_dir = current_dir.rsplit("/notebooks", 1)[0]
        else:
            # If not in notebooks directory, assume current directory is project root
            remote_dir = current_dir

        # Pull changes
        pull_result = remote_kernel.execute(f"!cd {remote_dir} && git pull")
        pull_details = []
        if pull_result and pull_result.get("outputs"):
            pull_details = [extract_output(output) for output in pull_result["outputs"]]

        return {
            "success": True,
            "message": "Changes pulled successfully",
            "current_dir": current_dir,
            "project_dir": remote_dir,
            "details": pull_details,
        }
    except Exception as e:
        return {"success": False, "message": f"Error pulling changes: {e!s}"}


@mcp.tool(description="Push changes from remote workspace to remote repository with commit")
async def push_remote_changes(
    project_name: str,
    commit_message: str,
    connection_info: dict,
) -> dict[str, Union[bool, str]]:
    """Push changes from remote workspace to remote repository with commit.

    Args:
        project_name: Name of the project in the remote workspace
        commit_message: Commit message for the changes
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing push status
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        remote_kernel.start()

        # Get current working directory using pwd
        pwd_output = remote_kernel.execute("!pwd")
        current_dir = ""
        if pwd_output and pwd_output.get("outputs"):
            current_dir = extract_output(pwd_output["outputs"][0]).strip()
        
        if not current_dir:
            return {"success": False, "message": "Could not determine current working directory"}

        # Determine project directory by going up from notebooks directory
        if "notebooks" in current_dir:
            # If we're in a notebooks directory, go up to find project root
            remote_dir = current_dir.rsplit("/notebooks", 1)[0]
        else:
            # If not in notebooks directory, assume current directory is project root
            remote_dir = current_dir

        print(f"Current directory: {current_dir}")
        print(f"Determined project directory: {remote_dir}")

        # Commit and push changes
        push_result = remote_kernel.execute(
            f"!cd {remote_dir} && git add . && git commit -m '{commit_message}' && git push"
        )

        return {
            "success": True,
            "message": "Changes committed and pushed successfully",
            "current_dir": current_dir,
            "project_dir": remote_dir,
        }
    except Exception as e:
        return {"success": False, "message": f"Error committing and pushing changes: {e!s}"}


@mcp.tool(description="Pull changes from remote repository to local workspace")
async def pull_remote_changes_on_local(
    local_dir: str,
    connection_info: dict,
) -> dict[str, Union[bool, str]]:
    """Pull changes from remote repository to local workspace.

    Args:
        local_dir: Absolute path to the local git repository
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing pull status
    """
    try:
        # Validate directory
        repo_path = Path(local_dir)
        if not repo_path.is_absolute():
            return {
                "success": False,
                "message": f"Directory must be absolute path, got: {local_dir}",
            }
        if not (repo_path / ".git").exists():
            return {
                "success": False,
                "message": f"Not a git repository: {local_dir}",
            }

        server_url, _ = extract_connection_info(connection_info)
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url) 
        )
        # Pull changes
        pull_output = remote_kernel.execute(f"!cd {local_dir} && git pull")

        return {
            "success": True,
            "message": "Changes pulled successfully",
            "details": [extract_output(output) for output in pull_output],
        }
    except Exception as e:
        return {"success": False, "message": f"Error pulling changes: {e!s}"}


@mcp.tool(description="Add a cell to a Jupyter notebook (WITHOUT execution)")
async def add_notebook_cell(
    connection_info: dict,
    cell_content: str,
) -> dict[str, Union[bool, int, str]]:
    """Add a code cell to a Jupyter notebook WITHOUT executing it.

    This tool creates a new code cell in the specified notebook and saves it
    permanently in the notebook file, but does NOT execute it. Use this for:

    WHEN TO USE:
    - Preparing code cells for later execution
    - Building notebook structure step by step
    - Adding code when you want to control execution timing
    - Batch adding multiple cells before execution

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)
        cell_content: Code content to add (WILL BE SAVED in notebook, NOT executed)

    Returns:
        dict containing cell addition status and cell index
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)

        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Add the cell without executing
            cell_index = notebook.add_code_cell(cell_content)

            return {
                "success": True,
                "cell_index": cell_index,
                "message": f"Code cell added at index {cell_index}",
            }
        finally:
            await notebook.stop()

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Execute an existing cell in a Jupyter notebook by index")
async def execute_existing_cell(
    connection_info: dict,
    cell_index: int,
) -> dict[str, Union[bool, list[str]]]:
    """Execute an existing cell in a Jupyter notebook by its index.

    This tool executes a cell that already exists in the notebook and returns
    the execution results. Use this for:

    WHEN TO USE:
    - Executing cells that were previously added
    - Re-running existing cells
    - Controlled execution of prepared code
    - Debugging specific cells

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)
        cell_index: Index of the cell to execute (0-based)

    Returns:
        dict containing execution status and outputs
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client (kernel is already running)
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        remote_kernel.start()

        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Check if cell index is valid
            ydoc = notebook._doc
            if cell_index >= len(ydoc._ycells):
                return {"success": False, "message": f"Cell index {cell_index} out of range"}

            # Execute the cell
            notebook.execute_cell(cell_index, remote_kernel)

            # Get outputs
            outputs = ydoc._ycells[cell_index]["outputs"]
            str_outputs = [extract_output(output) for output in outputs]

            # Check for errors
            has_error = any(output.get("output_type") == "error" for output in outputs)

            if has_error:
                error_output = next(
                    output for output in outputs if output.get("output_type") == "error"
                )
                error_info = {
                    "error_name": error_output.get("ename", "Unknown Error"),
                    "error_value": error_output.get("evalue", ""),
                    "traceback": error_output.get("traceback", []),
                }
                return {"success": False, "outputs": str_outputs, "error_info": error_info}

            return {"success": True, "outputs": str_outputs}
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Execute a cell in a Jupyter notebook (2-stage: add then execute)")
async def execute_notebook_cell(
    connection_info: dict,
    cell_content: str,
) -> dict[str, Union[bool, list[str]]]:
    """Execute a cell in a Jupyter notebook WITH notebook persistence using 2-stage approach.

    This tool creates a new cell in the specified notebook, executes it, and saves both
    the code and results permanently in the notebook file. This uses a 2-stage approach:
    1. Add the cell to the notebook
    2. Execute the cell and get results

    WHEN TO USE:
    - Building permanent analysis workflows
    - Creating reproducible data science notebooks
    - Documenting analysis steps with code and outputs
    - Sharing work that others need to reproduce
    - Creating tutorial or educational content

    WHEN NOT TO USE:
    - Quick debugging or one-off operations
    - System administration tasks
    - Code testing before finalizing
    - Operations that shouldn't be saved permanently

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)
        cell_content: Content of the cell to execute (WILL BE SAVED in notebook)

    Returns:
        dict containing execution status and outputs (SAVED to notebook)
    """
    try:
        # Stage 1: Add the cell
        add_result = await add_notebook_cell(connection_info, cell_content)
        if not add_result["success"]:
            return add_result

        cell_index = add_result["cell_index"]

        # Stage 2: Execute the cell
        execute_result = await execute_existing_cell(connection_info, cell_index)

        # Combine results
        if execute_result["success"]:
            return {
                "success": True,
                "outputs": execute_result["outputs"],
                "cell_index": cell_index,
            }
        else:
            # Add cell_index to error result
            execute_result["cell_index"] = cell_index
            return execute_result

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Update a specific cell in a Jupyter notebook")
async def update_notebook_cell(
    connection_info: dict,
    cell_index: int,
    new_content: str,
) -> dict[str, Union[bool, str]]:
    """Update the content of a specific cell in a Jupyter notebook.

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)
        cell_index: Index of the cell to update (0-based)
        new_content: New content for the cell

    Returns:
        dict containing update status
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)
        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Update the cell content
            ydoc = notebook._doc
            if cell_index < len(ydoc._ycells):
                ydoc._ycells[cell_index]["source"] = new_content
                return {"success": True, "message": "Cell updated successfully"}
            else:
                return {"success": False, "message": f"Cell index {cell_index} out of range"}
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Re-execute a specific cell in a Jupyter notebook")
async def rerun_notebook_cell(
    connection_info: dict,
    cell_index: int,
) -> dict[str, Union[bool, list[str]]]:
    """Re-execute a specific cell in a Jupyter notebook.

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)
        cell_index: Index of the cell to re-execute (0-based)

    Returns:
        dict containing execution status and outputs
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )

        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Check if cell index is valid
            ydoc = notebook._doc
            if cell_index >= len(ydoc._ycells):
                return {"success": False, "message": f"Cell index {cell_index} out of range"}

            # Re-execute the cell
            notebook.execute_cell(cell_index, remote_kernel)

            # Get outputs
            outputs = ydoc._ycells[cell_index]["outputs"]
            str_outputs = [extract_output(output) for output in outputs]

            # Check for errors
            has_error = any(output.get("output_type") == "error" for output in outputs)

            if has_error:
                error_output = next(
                    output for output in outputs if output.get("output_type") == "error"
                )
                error_info = {
                    "error_name": error_output.get("ename", "Unknown Error"),
                    "error_value": error_output.get("evalue", ""),
                    "traceback": error_output.get("traceback", []),
                }
                return {"success": False, "outputs": str_outputs, "error_info": error_info}

            return {"success": True, "outputs": str_outputs}
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(
    description="Monitor remote server resource usage including CPU, memory, disk, and YARN resources"
)
async def monitor_remote_resources(
    connection_info: dict,
) -> dict[str, Union[str, list[str]]]:
    """Monitor remote server resource usage.

    Args:
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing resource usage information
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url)
        )
        # Get system resources
        cpu_outputs = remote_kernel.execute("!top -bn1 | grep 'Cpu(s)' | awk '{print $2}'")
        cpu = [extract_output(output) for output in cpu_outputs]

        memory_outputs = remote_kernel.execute("!free -m | grep Mem | awk '{print $3/$2 * 100}'")
        memory = [extract_output(output) for output in memory_outputs]

        disk_outputs = remote_kernel.execute("!df -h | grep '/dev/sda1'")
        disk = [extract_output(output) for output in disk_outputs]

        # Get YARN resources
        yarn_outputs = remote_kernel.execute("!yarn node -list -showDetails")
        yarn = [extract_output(output) for output in yarn_outputs]

        return {
            "cpu_usage": cpu[0] if cpu else "N/A",
            "memory_usage_percent": memory[0] if memory else "N/A",
            "disk_usage": disk[0] if disk else "N/A",
            "yarn_nodes": yarn,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(description="Get all cells content from a Jupyter notebook")
async def get_notebook_cells(
    connection_info: dict,
) -> dict[str, Union[bool, list[dict]]]:
    """Get all cells content from a Jupyter notebook.

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)

    Returns:
        dict containing all notebook cells with their content and metadata
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)
        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Get all cells
            ydoc = notebook._doc
            cells = []

            for i, cell in enumerate(ydoc._ycells):
                cell_info = {
                    "index": i,
                    "type": cell.get("cell_type", "unknown"),
                    "source": cell.get("source", ""),
                }

                # Add execution count for code cells
                if cell.get("cell_type") == "code":
                    cell_info["execution_count"] = cell.get("execution_count")
                    # Include outputs if they exist
                    if cell.get("outputs"):
                        cell_info["has_outputs"] = True
                        cell_info["output_count"] = len(cell.get("outputs", []))

                cells.append(cell_info)

            return {"success": True, "cells": cells, "total_cells": len(cells)}
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Get the value of a variable from the kernel namespace")
async def get_variable(
    variable_name: str,
    connection_info: dict,
    mimetype: str = "text/plain",
) -> dict[str, Union[bool, str]]:
    """Get the value of a variable from the kernel namespace.

    Args:
        variable_name: Name of the variable to retrieve
        connection_info: Dict with jupyter connection details
        mimetype: MIME type for the output format (default: text/plain)

    Returns:
        dict containing the variable value and metadata
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        # Use the kernel client's get_variable method
        data, metadata = remote_kernel.get_variable(variable_name, mimetype)

        # Extract the value based on mimetype
        if mimetype in data:
            value = data[mimetype]
        else:
            # Fallback to text/plain if requested mimetype not available
            value = data.get("text/plain", str(data))

        return {
            "success": True,
            "variable_name": variable_name,
            "value": value,
            "mimetype": mimetype,
            "available_mimetypes": list(data.keys()),
            "metadata": metadata,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "variable_name": variable_name}


@mcp.tool(description="List all available variables in the kernel namespace")
async def list_variables(
    connection_info: dict,
) -> dict[str, Union[bool, list[dict]]]:
    """List all available variables in the kernel namespace.

    Args:
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing list of all variables with their descriptions
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )

        # Use the kernel client's list_variables method
        variable_descriptions = remote_kernel.list_variables()

        # Convert to dictionaries for better JSON serialization
        variables = []
        for var_desc in variable_descriptions:
            var_info = {
                "name": var_desc.name,
                "type": var_desc.type,
                "description": var_desc.description,
            }
            # Add size if available
            if hasattr(var_desc, "size") and var_desc.size is not None:
                var_info["size"] = var_desc.size
            variables.append(var_info)

        return {"success": True, "variables": variables, "total_variables": len(variables)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Execute code and inspect a specific variable in one operation")
async def execute_and_inspect(
    code: str,
    variable_name: str,
    connection_info: dict,
    mimetype: str = "text/plain",
) -> dict[str, Union[bool, str, list[str]]]:
    """Execute code and immediately inspect a specific variable.

    Args:
        code: Python code to execute
        variable_name: Name of the variable to inspect after execution
        connection_info: Dict with jupyter connection details
        mimetype: MIME type for the variable output format (default: text/plain)

    Returns:
        dict containing execution outputs and variable value
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        # Execute the code
        outputs = remote_kernel.execute(code)
        str_outputs = [extract_output(output) for output in outputs]

        # Check for execution errors
        has_error = any(output.get("output_type") == "error" for output in outputs)
        if has_error:
            error_output = next(
                output for output in outputs if output.get("output_type") == "error"
            )
            return {
                "success": False,
                "execution_outputs": str_outputs,
                "error": f"{error_output.get('ename', 'Error')}: {error_output.get('evalue', '')}",
            }

        # Get the variable value
        try:
            data, metadata = remote_kernel.get_variable(variable_name, mimetype)
            variable_value = data.get(mimetype, data.get("text/plain", str(data)))

            return {
                "success": True,
                "execution_outputs": str_outputs,
                "variable_name": variable_name,
                "variable_value": variable_value,
                "variable_mimetype": mimetype,
                "variable_metadata": metadata,
            }
        except Exception as var_error:
            return {
                "success": True,
                "execution_outputs": str_outputs,
                "variable_error": f"Failed to get variable '{variable_name}': {var_error}",
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Create a new notebook file in the project")
async def create_notebook(
    project_name: str,
    notebook_path: str,
    connection_info: dict,
) -> dict[str, Union[bool, str]]:
    """Create a new notebook file in the specified project.

    Args:
        project_name: Name of the project (e.g., 'my-data-project')
        notebook_path: Path to notebook relative to project_name/notebooks
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing creation status
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url)
        )
        # Ensure notebook directory exists
        notebook_dir = f"/workspace/{project_name}/notebooks/{os.path.dirname(notebook_path)}"
        remote_kernel.execute(f"!mkdir -p {notebook_dir}")

        full_path = f"/workspace/{project_name}/notebooks/{notebook_path}"

        # Connect to the notebook (this will create it if it doesn't exist)
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=full_path
            )
        )
        await notebook.start()

        try:
            # Add a basic welcome cell
            welcome_cell = (
                "# Welcome to the new notebook\n\n# This is an automatically created notebook"
            )
            notebook.add_code_cell(welcome_cell)

            return {"success": True, "message": "Notebook created successfully"}
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Save the current state of a notebook")
async def save_notebook(
    connection_info: dict,
) -> dict[str, Union[bool, str]]:
    """Save the current state of a notebook to disk.

    Args:
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)

    Returns:
        dict containing save status
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)
        # Connect to the notebook to trigger save
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # The notebook is automatically saved when we interact with it
            # We just need to ensure the connection is established
            ydoc = notebook._doc
            cell_count = len(ydoc._ycells) if ydoc._ycells else 0

            return {
                "success": True,
                "message": f"Notebook saved successfully with {cell_count} cells",
            }
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Delete cells from a notebook by index range")
async def delete_notebook_cells(
    start_index: int,
    connection_info: dict,
    end_index: int = None,
) -> dict[str, Union[bool, str]]:
    """Delete cells from a notebook by index range.

    Args:
        start_index: Starting cell index to delete (0-based)
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)
        end_index: Ending cell index to delete (inclusive), if None deletes only start_index

    Returns:
        dict containing deletion status
    """
    try:
        if end_index is None:
            end_index = start_index

        server_url, notebook_path = extract_connection_info(connection_info)
        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Delete cells in reverse order to maintain indices
            ydoc = notebook._doc
            total_cells = len(ydoc._ycells)

            if start_index >= total_cells or end_index >= total_cells:
                return {
                    "success": False,
                    "message": f"Cell index out of range. Notebook has {total_cells} cells",
                }

            for i in range(end_index, start_index - 1, -1):
                if i < len(ydoc._ycells):
                    del ydoc._ycells[i]

            deleted_count = end_index - start_index + 1
            return {
                "success": True,
                "message": f"Cells {start_index}-{end_index} deleted successfully",
            }
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Get kernel status and information")
async def get_kernel_status(connection_info: dict) -> dict[str, Union[bool, str]]:
    """Get the current kernel status and information.

    Args:
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing kernel status information
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        # Get kernel properties
        kernel_info = {
            "success": True,
            "kernel_id": remote_kernel.id,
            "execution_state": remote_kernel.execution_state,
            "is_alive": remote_kernel.is_alive(),
            "has_kernel": remote_kernel.has_kernel,
            "server_url": remote_kernel.server_url,
        }

        # Add optional properties if available
        if hasattr(remote_kernel, "last_activity") and remote_kernel.last_activity:
            kernel_info["last_activity"] = remote_kernel.last_activity

        if hasattr(remote_kernel, "username") and remote_kernel.username:
            kernel_info["username"] = remote_kernel.username

        if hasattr(remote_kernel, "kernel_info") and remote_kernel.kernel_info:
            kernel_info["kernel_info"] = remote_kernel.kernel_info

        return kernel_info
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Restart the remote kernel")
async def restart_kernel(connection_info: dict) -> dict[str, Union[bool, str]]:
    """Restart the remote kernel to clear state and memory.

    Args:
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing restart status
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )

        # Restart the kernel
        await remote_kernel.restart()

        return {"success": True, "message": "Kernel restarted successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Interrupt the current kernel execution")
async def interrupt_kernel(connection_info: dict) -> dict[str, Union[bool, str]]:
    """Interrupt the current kernel execution.

    Args:
        connection_info: Dict with jupyter connection details

    Returns:
        dict containing interrupt status
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        # Interrupt the kernel
        await remote_kernel.interrupt()

        return {"success": True, "message": "Kernel execution interrupted"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(
    description="Execute code with custom user expressions for detailed variable inspection"
)
async def execute_with_expressions(
    code: str,
    connection_info: dict,
    user_expressions: Optional[dict] = None,
    silent: bool = False,
) -> dict[str, Union[bool, str, list[str], dict]]:
    """Execute code with custom user expressions for detailed inspection.

    Args:
        code: Python code to execute
        connection_info: Dict with jupyter connection details
        user_expressions: Dict of expressions to evaluate (e.g., {"var_type": "type(var)"})
        silent: Whether to execute silently without adding to history

    Returns:
        dict containing execution results and evaluated expressions
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        remote_kernel = KernelClient(
            server_url=server_url, token=get_token_for_server(server_url), kernel_id=kernel_id
        )
        # Execute with user expressions using the kernel client's advanced execute method
        # Note: This uses the underlying execute method directly
        result = remote_kernel.execute(
            code,
            silent=silent,
            user_expressions=user_expressions or {},
        )

        # Extract outputs
        str_outputs = [
            extract_output(output) for output in result.get("content", {}).get("data", [])
        ]

        # Check for errors
        if result.get("content", {}).get("status") == "error":
            error_info = result.get("content", {})
            return {
                "success": False,
                "outputs": str_outputs,
                "error_info": {
                    "error_name": error_info.get("ename", "Unknown Error"),
                    "error_value": error_info.get("evalue", ""),
                    "traceback": error_info.get("traceback", []),
                },
            }

        # Extract user expressions results
        expressions_result = result.get("content", {}).get("user_expressions", {})

        return {
            "success": True,
            "outputs": str_outputs,
            "user_expressions": expressions_result,
            "execution_count": result.get("content", {}).get("execution_count"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Add a markdown cell to a Jupyter notebook (WITH notebook persistence)")
async def add_markdown_cell(
    markdown_content: str,
    connection_info: dict,
) -> dict[str, Union[bool, int, str]]:
    """Add a markdown cell to a Jupyter notebook WITH notebook persistence.

    This tool creates a new markdown cell in the specified notebook and saves it
    permanently in the notebook file. Use this for:

    WHEN TO USE:
    - Adding documentation and explanations to notebooks
    - Creating section headers and structure
    - Adding formatted text, links, and images
    - Creating tutorial or educational content
    - Documenting analysis steps and methodology

    Args:
        markdown_content: Markdown content to add (WILL BE SAVED in notebook)
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)

    Returns:
        dict containing cell addition status and cell index
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)
        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Add the markdown cell
            cell_index = notebook.add_markdown_cell(markdown_content)

            return {
                "success": True,
                "cell_index": cell_index,
                "message": "Markdown cell added successfully",
            }
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Add a raw cell to a Jupyter notebook (WITH notebook persistence)")
async def add_raw_cell(
    raw_content: str,
    connection_info: dict,
) -> dict[str, Union[bool, int, str]]:
    """Add a raw cell to a Jupyter notebook WITH notebook persistence.

    This tool creates a new raw cell in the specified notebook and saves it
    permanently in the notebook file. Use this for:

    WHEN TO USE:
    - Adding LaTeX content for PDF conversion
    - Including raw HTML for custom formatting
    - Adding content that should pass through unchanged
    - Creating content for specific output formats
    - Including formatted text that bypasses markdown processing

    Args:
        raw_content: Raw content to add (WILL BE SAVED in notebook)
        connection_info: Dict with jupyter connection details (notebookPath, kernelId, etc.)

    Returns:
        dict containing cell addition status and cell index
    """
    try:
        server_url, notebook_path = extract_connection_info(connection_info)
        # Connect to the notebook
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=server_url, token=get_token_for_server(server_url), path=notebook_path
            )
        )
        await notebook.start()

        try:
            # Add the raw cell
            cell_index = notebook.add_raw_cell(raw_content)

            return {
                "success": True,
                "cell_index": cell_index,
                "message": "Raw cell added successfully",
            }
        finally:
            await notebook.stop()
    except Exception as e:
        return {"success": False, "error": str(e)}


# EDA Resources - Templates and Documentation Only (users provide data paths)


@mcp.resource("templates://eda/{template_type}")
async def get_eda_template(template_type: str) -> str:
    """Get EDA code templates for common analysis patterns."""
    templates = {
        "basic": """# Basic EDA Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (replace with your actual data path)
df = pd.read_csv('your_dataset.csv')

# Basic info
print("Dataset shape:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())""",
        "visualization": """# Visualization Template
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogram
df['column_name'].hist(ax=axes[0,0], bins=30)
axes[0,0].set_title('Distribution')

# Box plot
df.boxplot(column='column_name', ax=axes[0,1])
axes[0,1].set_title('Box Plot')

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, ax=axes[1,0])
axes[1,0].set_title('Correlation Matrix')

# Scatter plot
df.plot.scatter(x='col1', y='col2', ax=axes[1,1])
axes[1,1].set_title('Scatter Plot')

plt.tight_layout()
plt.show()""",
        "statistical": """# Statistical Analysis Template
from scipy import stats
import pandas as pd

# Normality tests
def check_normality(data):
    statistic, p_value = stats.normaltest(data.dropna())
    return {'statistic': statistic, 'p_value': p_value, 'is_normal': p_value > 0.05}

# Correlation analysis
def correlation_analysis(df, target_col):
    correlations = df.corr()[target_col].sort_values(ascending=False)
    return correlations

# Group comparisons
def compare_groups(df, group_col, value_col):
    groups = [group for name, group in df.groupby(group_col)[value_col]]
    f_stat, p_value = stats.f_oneway(*groups)
    return {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}

# Example usage:
# normality_result = check_normality(df['column_name'])
# correlations = correlation_analysis(df, 'target_column')
# group_comparison = compare_groups(df, 'category', 'value')""",
        "missing_data": """# Missing Data Analysis Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_data(df):
    # Missing data summary
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    # Missing data heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
    plt.title('Missing Data Heatmap')
    plt.show()
    
    return missing_summary

# Usage:
# missing_analysis = analyze_missing_data(df)
# print(missing_analysis)""",
    }

    if template_type not in templates:
        available = ", ".join(templates.keys())
        return f"Template '{template_type}' not found. Available templates: {available}"

    return templates[template_type]


# Add static resources for documentation
@mcp.resource("docs://eda-guide")
async def eda_guide() -> str:
    """Comprehensive EDA workflow guide."""
    return """# EDA Workflow Guide

## 1. Data Preparation
- Ensure your dataset is accessible in the remote workspace
- Note the exact file path (e.g., '/workspace/project/data/dataset.csv')
- Verify file format is supported (.csv, .json, .parquet, .xlsx, .tsv)

## 2. Basic EDA Steps
1. Load and inspect data structure (shape, columns, dtypes)
2. Check for missing values and data quality issues
3. Generate summary statistics
4. Visualize distributions and relationships
5. Identify outliers and anomalies
6. Perform correlation analysis

## 3. Code Templates Available
- `templates://eda/basic` - Basic EDA workflow
- `templates://eda/visualization` - Common visualizations
- `templates://eda/statistical` - Statistical tests
- `templates://eda/missing_data` - Missing data analysis

## 4. Best Practices
- Always start with data.info() and data.describe()
- Check data types and convert if necessary
- Handle missing values appropriately
- Use appropriate visualizations for data types
- Document findings with markdown cells

## 5. Workflow Tools
- Use `auto_generate_eda_notebook()` with your data path for automated setup
- Use `get_eda_recommendations()` for AI-powered analysis guidance
- Use EDA prompts for structured workflow guidance
"""


# Update EDA guidance prompts to work with user-provided paths
@mcp.prompt()
def eda_workflow_guide(dataset_path: str, analysis_goal: str = "general exploration") -> str:
    """Generate a comprehensive EDA workflow prompt for a specific dataset path and goal."""
    return f"""# EDA Workflow for {dataset_path}

## Analysis Goal: {analysis_goal}

Please follow this structured EDA approach:

### Phase 1: Data Loading & Initial Inspection
1. Load the dataset from: `{dataset_path}`
2. Inspect basic properties:
   - Shape (rows, columns)
   - Column names and data types
   - Memory usage
   - First few rows

### Phase 2: Data Quality Assessment
1. Check for missing values (quantity and patterns)
2. Identify duplicate rows
3. Examine data types and potential conversion needs
4. Look for obvious data quality issues
5. Check for outliers and anomalies

### Phase 3: Descriptive Statistics
1. Generate summary statistics for numerical columns
2. Examine categorical variable frequencies
3. Understand data distributions
4. Identify potential data issues

### Phase 4: Data Visualization
1. Create distribution plots for key variables
2. Generate correlation heatmap for numerical variables
3. Create appropriate plots based on variable types
4. Look for patterns, trends, and relationships

### Phase 5: Deep Dive Analysis
Based on the analysis goal '{analysis_goal}', focus on:
- Relevant feature relationships
- Target variable analysis (if applicable)
- Segmentation opportunities
- Business insights

### Phase 6: Documentation
1. Use markdown cells to document findings
2. Summarize key insights and recommendations
3. Note data quality issues and suggested improvements

## Code Template Suggestions:
- Use `templates://eda/basic` for initial data exploration
- Use `templates://eda/visualization` for comprehensive plotting
- Use `templates://eda/statistical` for statistical analysis
- Use `templates://eda/missing_data` for missing data analysis

Remember to:
- Document each step with markdown explanations
- Use appropriate visualizations for each data type
- Keep the analysis goal in mind throughout
- Test hypotheses with statistical methods when appropriate
"""


@mcp.prompt()
def data_quality_check_prompt(columns: list[str]) -> str:
    """Generate a prompt for systematic data quality checking."""
    columns_str = ", ".join(columns)
    return f"""# Data Quality Assessment

Please systematically check the data quality for these columns: {columns_str}

For each column, examine:

## Numerical Columns:
- Range and distribution (min, max, mean, median)
- Presence of outliers (using IQR or z-score methods)
- Missing values and their patterns
- Unexpected values (negative numbers where they shouldn't be, etc.)

## Categorical Columns:
- Unique value counts and frequencies
- Presence of typos or inconsistent formatting
- Missing or null categories
- Unexpected categories

## Date/Time Columns:
- Date range and format consistency
- Invalid dates or future dates (if inappropriate)
- Missing timestamps
- Timezone considerations

## General Checks:
- Duplicate rows (exact and near-duplicates)
- Referential integrity (if applicable)
- Data consistency across related columns
- Completeness by row and column

Generate a comprehensive report with:
1. Summary of issues found
2. Severity assessment (critical, moderate, minor)
3. Recommended actions for each issue
4. Code to fix common problems

Use visualizations to support your findings where appropriate.
"""


@mcp.prompt()
def visualization_strategy_prompt(data_types: dict, target_variable: Optional[str] = None) -> str:
    """Generate a visualization strategy based on data types and analysis context."""
    target_info = (
        f"with target variable '{target_variable}'"
        if target_variable
        else "for exploratory analysis"
    )

    return f"""# Visualization Strategy {target_info}

Based on the data types in your dataset: {data_types}

## Recommended Visualization Approach:

### 1. Univariate Analysis
For each variable type, create appropriate single-variable visualizations:

**Numerical Variables:**
- Histograms for distribution shape
- Box plots for outlier detection
- Q-Q plots for normality assessment

**Categorical Variables:**
- Bar charts for frequency counts
- Pie charts for proportions (if few categories)

### 2. Bivariate Analysis
Explore relationships between variables:

**Numerical vs Numerical:**
- Scatter plots with correlation coefficients
- Joint plots for distribution + correlation

**Categorical vs Numerical:**
- Box plots grouped by category
- Violin plots for distribution comparison

**Categorical vs Categorical:**
- Stacked bar charts
- Mosaic plots for complex relationships

### 3. Multivariate Analysis
**Correlation Matrix:**
- Heatmap of all numerical correlations
- Identify highly correlated features

**Feature Relationships:**
- Pair plots for key variables
- Parallel coordinates for pattern detection

### 4. Target Variable Focus (if applicable):
{f'''
**Target Variable Analysis:**
- Distribution of {target_variable}
- Relationship with each feature
- Feature importance visualization
''' if target_variable else ''}

### 5. Advanced Visualizations:
- Principal Component Analysis (PCA) plots
- t-SNE for high-dimensional data visualization
- Feature importance plots (if doing modeling)

## Implementation Notes:
- Use consistent color schemes
- Include proper titles and axis labels
- Add statistical annotations where relevant
- Create a dashboard-style summary at the end

Would you like me to implement any specific visualization from this strategy?
"""


@mcp.prompt()
def statistical_analysis_guide(analysis_type: str, variables: list[str]) -> str:
    """Guide for performing statistical analysis on specific variables."""
    vars_str = ", ".join(variables)

    return f"""# Statistical Analysis Guide: {analysis_type}

Variables to analyze: {vars_str}

## Analysis Approach for {analysis_type}:

### 1. Hypothesis Formation
Before testing, clearly state:
- Null hypothesis (H₀)
- Alternative hypothesis (H₁)
- Significance level (typically α = 0.05)

### 2. Test Selection Matrix:

**For Normality Testing:**
- Shapiro-Wilk test (small samples < 5000)
- Anderson-Darling test (medium samples)
- Kolmogorov-Smirnov test (large samples)

**For Correlation Analysis:**
- Pearson correlation (linear relationships, normal data)
- Spearman correlation (monotonic relationships, non-parametric)
- Kendall's tau (robust to outliers)

**For Group Comparisons:**
- Independent t-test (2 groups, normal data)
- Mann-Whitney U (2 groups, non-parametric)
- ANOVA (3+ groups, normal data)
- Kruskal-Wallis (3+ groups, non-parametric)

**For Association Testing:**
- Chi-square test (categorical independence)
- Fisher's exact test (small sample categories)

### 3. Assumptions Checking:
Before applying tests, verify:
- Data distribution (normality where required)
- Independence of observations
- Homogeneity of variances
- Sample size requirements

### 4. Effect Size Calculation:
Along with p-values, calculate:
- Cohen's d (for t-tests)
- Eta-squared (for ANOVA)
- Cramér's V (for chi-square)

### 5. Interpretation Guidelines:
- Statistical significance vs practical significance
- Confidence intervals
- Power analysis considerations
- Multiple comparison corrections (if applicable)

### 6. Reporting Template:
For each test, report:
- Test statistic and p-value
- Effect size and confidence intervals
- Assumption check results
- Practical interpretation of findings

Would you like me to implement any specific statistical test from this guide?
"""


@mcp.tool()
async def get_eda_recommendations(dataset_info: str, analysis_goal: str) -> dict:
    """Get AI-powered recommendations for EDA approach based on dataset characteristics."""
    try:
        # Use simple recommendations based on common patterns
        recommendations = f"""
# EDA Recommendations for: {analysis_goal}

Based on the dataset information provided, here are specific recommendations:

## Priority Areas:
1. **Data Quality Assessment**: Check for missing values, duplicates, and data type consistency
2. **Distribution Analysis**: Examine the distribution of key variables
3. **Correlation Analysis**: Identify relationships between variables
4. **Outlier Detection**: Look for unusual values that might indicate data quality issues

## Recommended Visualizations:
- Histograms for numerical variable distributions
- Box plots for outlier detection
- Correlation heatmap for feature relationships
- Bar charts for categorical variable frequencies

## Statistical Analysis:
- Descriptive statistics for all variables
- Normality tests for numerical variables
- Chi-square tests for categorical associations
- Correlation tests for numerical relationships

## Next Steps:
1. Start with basic data loading and inspection
2. Create visualizations for key variables
3. Perform statistical tests based on your analysis goals
4. Document findings and insights

Dataset Context: {dataset_info}
Analysis Goal: {analysis_goal}
"""

        return {
            "success": True,
            "recommendations": recommendations,
            "dataset_context": dataset_info,
            "analysis_goal": analysis_goal,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def auto_generate_eda_notebook(
    dataset_path: str,
    project_name: str,
    analysis_goal: str,
    connection_info: dict,
) -> dict:
    """Automatically generate a complete EDA notebook structure with guided analysis."""
    try:
        server_url, _ = extract_connection_info(connection_info)
        # Create the notebook
        notebook_name = f"eda_{dataset_path.replace('.', '_').replace('/', '_')}.ipynb"
        notebook_path = f"auto_eda/{notebook_name}"

        # Create notebook
        create_result = await create_notebook(project_name, notebook_path, connection_info)
        if not create_result["success"]:
            return create_result

        # Add introduction markdown
        intro_content = f"""# Automated EDA: {dataset_path}

## Analysis Objective
{analysis_goal}

## Dataset Information
- **File**: {dataset_path}
- **Analysis Goal**: {analysis_goal}
- **Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This notebook was automatically generated to guide your exploratory data analysis.
"""

        await add_markdown_cell(intro_content, connection_info)

        # Add data loading section
        loading_code = f"""# Data Loading and Initial Inspection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Find the dataset in the remote workspace
workspace = '/workspace'
dataset_filename = '{dataset_path}'

# Search for the dataset file
dataset_full_path = None
for root, dirs, files in os.walk(workspace):
    if dataset_filename in files:
        dataset_full_path = os.path.join(root, dataset_filename)
        break

if dataset_full_path:
    print(f"Found dataset at: {{dataset_full_path}}")
    
    # Load based on file extension
    file_ext = dataset_filename.split('.')[-1].lower()
    
    if file_ext == 'csv':
        df = pd.read_csv(dataset_full_path)
    elif file_ext == 'json':
        df = pd.read_json(dataset_full_path)
    elif file_ext == 'parquet':
        df = pd.read_parquet(dataset_full_path)
    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(dataset_full_path)
    elif file_ext == 'tsv':
        df = pd.read_csv(dataset_full_path, sep='\\t')
    else:
        # Try CSV as fallback
        df = pd.read_csv(dataset_full_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {{df.shape}}")
    print(f"Columns: {{list(df.columns)}}")
else:
    print(f"ERROR: Dataset '{{dataset_filename}}' not found in workspace '{{workspace}}'")
    print("Available datasets:")
    for root, dirs, files in os.walk(workspace):
        for file in files:
            if file.endswith(('.csv', '.json', '.parquet', '.xlsx', '.tsv')):
                print(f"  - {{file}} ({{os.path.join(root, file)}})")
"""

        await execute_notebook_cell(connection_info, loading_code)

        # Add basic inspection section
        inspection_code = """# Basic Data Inspection
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"\\nData Types:")
print(df.dtypes)
print(f"\\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\\n=== FIRST 5 ROWS ===")
display(df.head())

print("\\n=== SUMMARY STATISTICS ===")
display(df.describe())

print("\\n=== MISSING VALUES ===")
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]
if len(missing_summary) > 0:
    print("Columns with missing values:")
    print(missing_summary)
else:
    print("No missing values found!")
"""

        await execute_notebook_cell(connection_info, inspection_code)

        # Add visualization section placeholder
        viz_placeholder = """## Data Visualization Section

This section will contain visualizations based on your data types and analysis goals.

**Recommended next steps:**
1. Run the cells above to understand your data structure
2. Use the `get_eda_recommendations` tool with your dataset info
3. Use `templates://eda/visualization` resource for visualization templates
4. Customize the analysis based on your specific needs

**Quick visualization starter:**"""

        await add_markdown_cell(viz_placeholder, connection_info)

        # Add basic visualization code
        basic_viz_code = """# Quick Data Overview Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Dataset Overview Dashboard', fontsize=16)

# 1. Missing data heatmap (if any missing data)
if df.isnull().sum().sum() > 0:
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=axes[0,0])
    axes[0,0].set_title('Missing Data Pattern')
else:
    axes[0,0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=axes[0,0].transAxes)
    axes[0,0].set_title('Missing Data: None')

# 2. Numeric columns distribution
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    df[numeric_cols].hist(ax=axes[0,1], bins=20)
    axes[0,1].set_title(f'Numeric Distributions ({len(numeric_cols)} cols)')
else:
    axes[0,1].text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center', transform=axes[0,1].transAxes)

# 3. Correlation heatmap (for numeric columns)
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
    axes[1,0].set_title('Correlation Matrix')
else:
    axes[1,0].text(0.5, 0.5, 'Need 2+ Numeric Columns', ha='center', va='center', transform=axes[1,0].transAxes)

# 4. Data types distribution
dtype_counts = df.dtypes.value_counts()
axes[1,1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
axes[1,1].set_title('Data Types Distribution')

plt.tight_layout()
plt.show()

print("\\n=== QUICK INSIGHTS ===")
print(f"• Total columns: {len(df.columns)}")
print(f"• Numeric columns: {len(numeric_cols)}")
print(f"• Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
print(f"• Missing values: {df.isnull().sum().sum()}")
print(f"• Duplicate rows: {df.duplicated().sum()}")
"""

        await execute_notebook_cell(connection_info, basic_viz_code)

        return {
            "success": True,
            "notebook_path": notebook_path,
            "message": f"Generated comprehensive EDA notebook at {notebook_path}",
            "next_steps": [
                "Review the generated analysis",
                "Use get_eda_recommendations tool for specific guidance",
                "Customize analysis based on your domain knowledge",
                "Use EDA templates for additional analysis patterns",
            ],
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Debug tool: execute command and return raw output for debugging")
async def debug_execute_command(
    connection_info: dict,
    command: str,
    execution_type: str = "shell",
) -> dict:
    """Debug tool to execute commands and return raw unprocessed output.

    This tool is designed for debugging purposes to see the raw output structure
    from kernel execution without any processing.

    Args:
        connection_info: Dict with jupyter connection details
        command: The command to execute
        execution_type: Either "shell" or "python"

    Returns:
        dict containing raw execution results for debugging
    """
    try:
        server_url, _ = extract_connection_info(connection_info)
        kernel_id = connection_info.get("kernelId")

        # Initialize remote kernel client
        try:
            token = get_token_for_server(server_url)
        except Exception as token_error:
            return {
                "success": False,
                "error": f"Token selection failed: {token_error}",
                "debug_info": "Token selection stage"
            }

        remote_kernel = KernelClient(server_url=server_url, token=token, kernel_id=kernel_id)
        remote_kernel.start()

        if execution_type == "shell":
            # Execute shell command with ! prefix
            result = remote_kernel.execute(f"!{command}")
        elif execution_type == "python":
            # Execute Python code directly
            result = remote_kernel.execute(command)
        else:
            return {
                "success": False,
                "error": f"Invalid execution_type: {execution_type}. Use 'shell' or 'python'",
            }

        # Return completely raw result for debugging
        return {
            "success": True,
            "command": command,
            "execution_type": execution_type,
            "raw_result": result,
            "raw_result_type": type(result).__name__,
            "raw_result_keys": list(result.keys()) if isinstance(result, dict) else "Not a dict",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "debug_info": f"Exception in debug_execute_command: {type(e).__name__}"
        }


if __name__ == "__main__":
    # Run the server with STDIO transport (default)
    mcp.run()
