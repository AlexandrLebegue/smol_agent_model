from typing import Any, Optional
from smolagents.tools import Tool
import os
import sys

class ModifyFileTool(Tool):
    name = "modify_file"
    description = "Modify an existing file."
    inputs = {
        'path': {'type': 'string', 'description': 'The path of the file to modify.'},
        'content': {'type': 'string', 'description': 'The new content of the file.'},
        'append': {'type': 'boolean', 'description': 'If true, append the content to the end of the file instead of replacing it.', 'nullable': True}
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__()
        # Detect the operating system
        self.is_windows = sys.platform.startswith('win')

    def forward(self, path: str, content: str, append: bool = False) -> str:
        try:
            # Check if the file exists
            if not os.path.exists(path):
                return f"Error: The file {path} does not exist."
            
            # Open mode based on the append option
            mode = 'a' if append else 'w'
            
            # Modify the file
            with open(path, mode, encoding='utf-8') as file:
                file.write(content)
                
            action = "modified" if not append else "updated (content added)"
            return f"File {action} successfully: {path}"
        except Exception as e:
            return f"Error while modifying the file: {str(e)}" 