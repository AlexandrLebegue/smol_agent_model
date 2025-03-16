from typing import Any, Optional
from smolagents.tools import Tool
import os

class ValidateFinalAnswer(Tool):
    name = "validate_final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem to be validate'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        try:
            compile(answer, "bogusfile.py", "exec")
            # os.remove("bogusfile.py")
            return "Answer is valide and can be submitted to final answer."
        except Exception as e:
            return f"Invalid answer : {e}"


    def __init__(self, *args, **kwargs):
        self.is_initialized = False
