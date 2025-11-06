from huggingface_hub import HfApi
from pathlib import Path

api = HfApi(token="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")

folder_path = Path("output")
repo_id = "G7xHp2Qv/VirDA"

# Find all .pth files while preserving folder structure
pth_files = list(folder_path.rglob("*.pth"))

print(f"Found {len(pth_files)} .pth files to upload")

# Upload each .pth file, preserving the folder structure
for pth_file in pth_files:
    # Get relative path from the folder_path to preserve structure
    relative_path = pth_file.relative_to(folder_path)
    path_in_repo = str(relative_path)
    
    print(f"Uploading: {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(pth_file),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )

print("Checkpoints uploaded successfully!")