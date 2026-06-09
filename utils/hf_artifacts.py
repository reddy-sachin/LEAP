from huggingface_hub import hf_hub_download


DEFAULT_DATASET_REPO_ID = "reddysachin/LEAP_dataset"
DEFAULT_MODEL_REPO_ID = "reddysachin/LEAP"


def _download_hf_file(repo_id, repo_type, filename):
    """Download one file from Hugging Face and print progress.

    Args:
        repo_id: Hugging Face repo id (`user/repo`).
        repo_type: One of `dataset` or `model`.
        filename: File name inside the repo.

    Returns:
        Local cache path to the downloaded file.
    """
    print(f"[HF] Downloading {filename} ...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    print(f"[HF] Ready: {filename}")
    return path


def download_dataset_files(repo_id=DEFAULT_DATASET_REPO_ID):
    """Download train/validation/test dataset splits from Hugging Face.

    Args:
        repo_id: Dataset repo id.

    Returns:
        Dict mapping split names to local cached file paths.
    """
    repo_type = "dataset"
    files = [
        ("train", "train.parquet"),
        ("validation", "validation.parquet"),
        ("test", "test.parquet"),
    ]
    out = {}
    print(f"[HF] Dataset repo: {repo_id}")
    for key, filename in files:
        out[key] = _download_hf_file(repo_id=repo_id, repo_type=repo_type, filename=filename)
    return out


def download_dataset_test_file(repo_id=DEFAULT_DATASET_REPO_ID):
    """Download only the test split from the dataset repo.

    Args:
        repo_id: Dataset repo id.

    Returns:
        Local cache path to `test.parquet`.
    """
    repo_type = "dataset"
    filename = "test.parquet"
    print(f"[HF] Dataset repo: {repo_id}")
    return _download_hf_file(repo_id=repo_id, repo_type=repo_type, filename=filename)


def download_model_artifacts(repo_id=DEFAULT_MODEL_REPO_ID):
    """Download pretrained model artifacts from Hugging Face.

    Args:
        repo_id: Model repo id.

    Returns:
        Dict with local cache paths for `weights` and `scaler`.
    """
    repo_type = "model"
    print(f"[HF] Model repo: {repo_id}")
    weights = _download_hf_file(
        repo_id=repo_id,
        repo_type=repo_type,
        filename="leap_weights.pt",
    )
    scaler = _download_hf_file(
        repo_id=repo_id,
        repo_type=repo_type,
        filename="leap_scaler.pkl",
    )
    print("[HF] Model artifacts ready")
    return {"weights": weights, "scaler": scaler}


def download_flybys_csv(repo_id="reddysachin/Europa_Clipper_Tour", filename="clipper_tour_with_BJ.csv"):
    """Download Europa Clipper flyby trajectory CSV from Hugging Face.

    Args:
        repo_id: Dataset repo id.
        filename: CSV filename in the repo.

    Returns:
        Local cache path to the downloaded CSV file.
    """
    repo_type = "dataset"
    print(f"[HF] Flybys repo: {repo_id}")
    csv_path = _download_hf_file(repo_id=repo_id, repo_type=repo_type, filename=filename)
    print(f"[HF] Flybys CSV ready")
    return csv_path
