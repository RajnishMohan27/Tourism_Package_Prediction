
"""
Hosting Script
Uploads deployment files to the Hugging Face Space.
"""

import os

from huggingface_hub import login
from huggingface_hub import upload_file

# ------------------------------------------------------------
# Authenticate
# ------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise ValueError(
        "HF_TOKEN environment variable not found."
    )

# ------------------------------------------------------------
# Hugging Face Space
# ------------------------------------------------------------

#HF_SPACE_REPO = "RajnishMohan/tourism-package-prediction-app"
HF_SPACE_REPO = "RajnishMohan/tourism-package-prediction"

FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "Dockerfile"
]

# ------------------------------------------------------------
# Upload Deployment Files
# ------------------------------------------------------------

for file_name in FILES_TO_UPLOAD:

    print(f"Uploading {file_name}...")

    upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=HF_SPACE_REPO,
        repo_type="space"
    )

# ------------------------------------------------------------
# Upload Hugging Face README
# ------------------------------------------------------------

print("Uploading README_HF.md as README.md...")

upload_file(
    path_or_fileobj="README_HF.md",
    path_in_repo="README.md",
    repo_id=HF_SPACE_REPO,
    repo_type="space"
)

print("Deployment completed successfully.")
