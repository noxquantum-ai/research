"""Nox Model Export & Artifact Synchronization Utility.

Handles exporting compressed model checkpoints from cloud instances (RunPod / Lambda)
via HuggingFace Hub upload, direct tarball packaging, or secure transfer links.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.getenv("HF_TOKEN")


def log_step(msg):
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def package_tarball(checkpoint_dir, output_path="nox_checkpoints_master.tar.gz"):
    """
    Compresses checkpoint directory into a single optimized tarball archive.
    """
    log_step(f"Packaging '{checkpoint_dir}' into '{output_path}'...")
    cmd = ["tar", "-czf", output_path, "-C", str(Path(checkpoint_dir).parent), str(Path(checkpoint_dir).name)]
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log_step(f"Archive created: '{output_path}' ({size_mb:.2f} MB)")
    return output_path, size_mb


def upload_to_huggingface(checkpoint_dir, repo_id, token=None, private=True):
    """
    Uploads compressed checkpoints directly to a Hugging Face model repository.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        log_step("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi, create_repo

    auth_token = token or HF_TOKEN
    if not auth_token:
        raise ValueError("Hugging Face API token is required. Pass --hf_token or set HF_TOKEN in environment.")

    api = HfApi(token=auth_token)
    log_step(f"Verifying / creating Hugging Face repository '{repo_id}' (private={private})...")
    create_repo(repo_id=repo_id, token=auth_token, private=private, repo_type="model", exist_ok=True)

    log_step(f"Uploading '{checkpoint_dir}' to Hugging Face '{repo_id}'...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    log_step(f"Upload complete! View your model at: https://huggingface.co/{repo_id}")


def upload_to_transfersh(tarball_path):
    """
    Uploads tarball to transfer.sh for a direct one-click browser download link.
    """
    log_step(f"Uploading '{tarball_path}' to transfer service...")
    cmd = f"curl --progress-bar --upload-file {tarball_path} https://transfer.sh/{os.path.basename(tarball_path)}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        download_url = res.stdout.strip()
        log_step(f"Direct download link: {download_url}")
        return download_url
    else:
        log_step(f"Transfer upload failed: {res.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export Nox compressed checkpoints from cloud instance")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/day3_day4_cascade/master_cliff_sweep",
        help="Path to directory containing compressed checkpoints",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hf", "package", "transfer", "all"],
        default="all",
        help="Export method: 'hf' (HuggingFace Hub), 'package' (tarball), 'transfer' (one-click URL), or 'all'",
    )
    parser.add_argument("--hf_repo", type=str, default=None, help="Target HuggingFace repo (e.g. username/nox-llama-2-7b)")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--public", action="store_true", help="Make HuggingFace repository public (default: private)")
    parser.add_argument("--output_tar", type=str, default="nox_checkpoints_master.tar.gz", help="Output tarball name")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: '{args.checkpoint_dir}'")

    print("\n" + "=" * 90)
    print(" NOX MODEL EXPORT & ARTIFACT SYNCHRONIZATION")
    print("=" * 90)
    print(f"Source Directory: {args.checkpoint_dir}")

    # Method 1: Package tarball
    if args.method in ["package", "transfer", "all"]:
        tarball_path, size_mb = package_tarball(args.checkpoint_dir, args.output_tar)
        print(f"\n[Tarball Ready]: {tarball_path} ({size_mb:.2f} MB)")
        print("To download via RunPod Web UI:")
        print(f"  1. Navigate to the RunPod File Manager / Jupyter workspace.")
        print(f"  2. Right-click '{tarball_path}' and select 'Download'.")
        print("\nTo download via SSH / SCP on your local machine:")
        print(f"  scp -P <PORT> root@<RUNPOD_IP>:/workspace/research/{tarball_path} .")

    # Method 2: One-click transfer link
    if args.method in ["transfer", "all"]:
        print("\nGenerating direct one-click browser download link...")
        url = upload_to_transfersh(args.output_tar)
        if url:
            print(f"\n[One-Click Download Link]: {url}")
            print("You can paste this URL into your laptop browser to download immediately.")

    # Method 3: HuggingFace Model Hub
    if args.method in ["hf", "all"] and (args.hf_repo or os.getenv("HF_REPO")):
        repo = args.hf_repo or os.getenv("HF_REPO")
        print(f"\nUploading to Hugging Face Model Hub: '{repo}'...")
        upload_to_huggingface(
            checkpoint_dir=args.checkpoint_dir,
            repo_id=repo,
            token=args.hf_token,
            private=not args.public,
        )

    print("\n" + "=" * 90)
    print(" EXPORT PIPELINE COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
