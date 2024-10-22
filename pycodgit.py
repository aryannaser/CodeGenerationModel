from github import Github
import re
from datasets import Dataset
import os

# initialize PyGithub with the GitHub token
g = os.getenv('GithubToken')

# specify the repositorygit 
repo = g.get_repo("openai/gym")

# function to extract Python functions from a script
def extract_functions_from_code(code):
    pattern = re.compile(r"def\s+(\w+)\s*\(.*\):")
    functions = pattern.findall(code)
    return functions

# fetch Python files from the repository
python_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    elif file_content.path.endswith(".py"):
        python_files.append(file_content)

# extract functions and create dataset
data = {"code": [], "function_name": []}
for file in python_files:
    code = file.decoded_content.decode("utf-8")
    functions = extract_functions_from_code(code)
    for function in functions:
        data["code"].append(code)
        data["function_name"].append(function)

# create a Hugging Face dataset
dataset = Dataset.from_dict(data)

# save the dataset to disk
dataset.save_to_disk("code_generation_dataset")

print("Dataset created and saved to disk.")