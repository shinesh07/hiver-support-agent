#!/usr/bin/env python3
"""
Download script for the Twitter Customer Support dataset.

This script fetches the 'Customer Support on Twitter' dataset from Kaggle
(thoughtvector/customer-support-on-twitter) and stores the raw files in data/raw/.

To use automated download:
  1. Create a Kaggle account and generate an API token (kaggle.json).
  2. Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file or environment.
  3. Run: python scripts/download.py

Alternatively, you can manually download `twcs.csv` from:
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
and place it directly into the `data/raw/` directory.
"""

import os
import sys
from pathlib import Path
import click
from dotenv import load_dotenv

# Dataset identifier on Kaggle
KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
DATA_RAW_DIR = Path("data/raw")


def print_directory_summary(target_dir: Path) -> None:
    """Print a summary of files located in the target directory."""
    if not target_dir.exists():
        click.echo(f"Directory {target_dir} does not exist.")
        return

    files = list(target_dir.glob("*"))
    if not files:
        click.echo(f"\n[INFO] {target_dir} is currently empty.")
        return

    click.echo(f"\n================ Files in {target_dir} ================")
    total_size = 0
    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            size_mb = size / (1024 * 1024)
            click.echo(f" - {f.name} ({size_mb:.2f} MB)")
        elif f.is_dir():
            click.echo(f" - [DIR] {f.name}/")

    total_mb = total_size / (1024 * 1024)
    click.echo(f"Total size: {total_mb:.2f} MB ({len(files)} items)")
    click.echo("======================================================")


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DATA_RAW_DIR,
    help="Target directory for downloaded raw files (default: data/raw).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-download even if files already exist.",
)
def download(output_dir: Path, force: bool) -> None:
    """Download the Twitter Customer Support dataset into data/raw/."""
    # Load environment variables from .env if present
    load_dotenv()

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_file = output_dir / "twcs.csv"

    if expected_file.exists() and not force:
        click.secho(
            f"[OK] Raw dataset already exists at {expected_file}. Skipping download.",
            fg="green",
        )
        print_directory_summary(output_dir)
        return

    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    click.echo(f"Target raw directory: {output_dir.resolve()}")

    if not kaggle_username or not kaggle_key:
        click.secho(
            "\n[WARNING] Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY) are not set in environment or .env.",
            fg="yellow",
        )
        click.echo(
            "To download automatically:\n"
            "  1. Export KAGGLE_USERNAME and KAGGLE_KEY or set them in .env.\n"
            "  2. Re-run: python scripts/download.py\n\n"
            "Manual Download Instructions:\n"
            f"  1. Go to: https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"  2. Download twcs.csv.zip, extract it, and copy 'twcs.csv' to '{output_dir}/twcs.csv'.\n"
        )
        print_directory_summary(output_dir)
        return

    # If Kaggle credentials are configured, attempt download via kaggle API
    click.echo(f"Attempting download for {KAGGLE_DATASET} via Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            KAGGLE_DATASET, path=str(output_dir), unzip=True, quiet=False
        )
        click.secho("\n[SUCCESS] Download completed successfully.", fg="green")
    except ImportError:
        click.secho(
            "\n[NOTICE] 'kaggle' package is not installed. You can install it with `pip install kaggle`\n"
            f"or manually download the dataset from https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and place 'twcs.csv' into '{output_dir}'.",
            fg="yellow",
        )
    except Exception as e:
        click.secho(f"\n[ERROR] Failed to download dataset via Kaggle API: {e}", fg="red")
        click.echo(
            f"Please download manually from https://www.kaggle.com/datasets/{KAGGLE_DATASET} "
            f"and place 'twcs.csv' in '{output_dir}'."
        )

    print_directory_summary(output_dir)


if __name__ == "__main__":
    download()
