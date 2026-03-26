import os
from smolagents import CodeAgent, HfApiModel, tool, DuckDuckGoSearchTool
import tempfile
import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

@tool
def download_task_file(task_id: str) -> str:
    """
    Downloads the file associated with a given task ID.
    
    Args:
        task_id: The ID of the task to download the file for.
        
    Returns:
        The local path to the downloaded file, or a message indicating no file or error.
    """
    url = f"{DEFAULT_API_URL}/files/{task_id}"
    try:
        response = requests.get(url)
        if response.status_code == 404:
            return "No file found for this task_id."
        response.raise_for_status()
        
        # Check disposition for filename
        cd = response.headers.get("content-disposition")
        filename = "downloaded_file"
        if cd and "filename=" in cd:
            import re
            m = re.search(r"filename=(.+)", cd)
            if m:
                filename = m.group(1).strip("\"'")
        
        filepath = os.path.join(tempfile.gettempdir(), filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return f"File formally downloaded successfully to {filepath}."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

class AdvancedAgent:
    def __init__(self):
        self.model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
        self.agent = CodeAgent(
            model=self.model,
            tools=[DuckDuckGoSearchTool(), download_task_file],
            additional_authorized_imports=["pandas", "numpy", "Pillow", "pytesseract", "json", "re", "math", "datetime", "PyPDF2"],
            max_steps=8
        )
        self.sys_prompt = """You are an expert agent solving GAIA level 1 questions. 
When asked a question, solve it by leveraging your code interpreter and tools. 
IMPORTANT: Your final answer MUST be EXACTLY the answer, with NO prefix, NO introductory words. 
Do not write "The answer is" or "Final Answer:". Just write the value or result.
If the question is about a file, immediately call `download_task_file` with the provided task ID.
"""

    def __call__(self, question: str, task_id: str = None) -> str:
        prompt = self.sys_prompt + f"\n\nQuestion: {question}"
        if task_id:
            prompt += f"\nAssociated task_id: {task_id}"
            
        try:
            result = self.agent.run(prompt)
            # Remove any trailing / leading spaces or quotes
            return str(result).strip(' "\'')
        except Exception as e:
            print(f"Agent error: {e}")
            return "Error"

if __name__ == "__main__":
    agent = AdvancedAgent()
    print(agent("What is 2+2?", "test-task"))
