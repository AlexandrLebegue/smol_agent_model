from typing import Any, Optional
from smolagents.tools import Tool
import os
import sys

class CreateFileTool(Tool):
    name = "create_file"
    description = "Creates a new file with the specified content."
    inputs = {
        'path': {'type': 'string', 'description': 'The path of the file to create.'},
        'content': {'type': 'string', 'description': 'The content to write in the file.'}
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__()
        # Detect the operating system
        self.is_windows = sys.platform.startswith('win')

    def forward(self, path: str, content: str) -> str:
        try:
            # Ensure the parent directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write the content to the file
            with open(path, 'w', encoding='utf-8') as file:
                file.write(content)
                
            return f"File created successfully: {path}"
        except Exception as e:
            return f"Error while creating the file: {str(e)}" 