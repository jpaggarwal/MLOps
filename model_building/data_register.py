from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "jpaggarwal/Bank-Customer-Churn"
repo_type = "dataset"
folder_path = "mlops/data"

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN is missing. Add it in GitHub Secrets.")

if not os.path.exists(folder_path):
    raise FileNotFoundError(f"Folder not found: {folder_path}")

api = HfApi(token=hf_token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=hf_token
    )
    print(f"Dataset repository '{repo_id}' created.")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    token=hf_token
)

print("Dataset uploaded successfully.")
