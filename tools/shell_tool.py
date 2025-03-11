from typing import Any, Optional
from smolagents.tools import Tool
import duckduckgo_search
import subprocess
import sys

class ShellCommandTool(Tool):
    name = "shell_command"
    description = "Executes a shell command and returns the result."
    inputs = {'command': {'type': 'string', 'description': 'The shell command to execute.'}}
    output_type = "string"

    def __init__(self, timeout=60, **kwargs):
        super().__init__()
        self.timeout = timeout
        # Detect the operating system
        self.is_windows = sys.platform.startswith('win')

    def forward(self, command: str) -> str:
        try:
            # Use shell=True for complex commands
            # Use different configurations based on the OS
            if self.is_windows:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=self.timeout,
                    executable="cmd.exe" if self.is_windows else None
                )
            else:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=self.timeout
                )
            
            # Return stdout and stderr if present
            result = ""
            if process.stdout:
                result += f"## Standard Output\n```\n{process.stdout}\n```\n\n"
            if process.stderr:
                result += f"## Standard Error\n```\n{process.stderr}\n```\n\n"
            
            result += f"Exit Code: {process.returncode}"
            return result
            
        except subprocess.TimeoutExpired:
            return "The command has timed out."
        except Exception as e:
            return f"Error while executing the command: {str(e)}"
