"""Publish the fine-tuned model to Hugging Face."""

from huggingface_hub import HfApi

REPO_ID = "n10046793/lab09-bike-share-ticket-router"


def main():
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path="model",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"published to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
