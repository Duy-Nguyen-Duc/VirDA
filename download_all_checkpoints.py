import os
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "G7xHp2Qv/ViRDA"

output_dir = f"{os.getcwd()}/checkpoints"
os.makedirs(output_dir, exist_ok=True)

token="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"

repo_files = list_repo_files(repo_id=repo_id, repo_type="model", token=token)
# Download each file from the repository
for file_path in repo_files:
    if (file_path.endswith(".pth")):
        local_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="model",
            local_dir=output_dir,
            force_download=True,
            token = token
        )

print("All checkpoint files downloaded successfully!")
