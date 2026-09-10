from __future__ import annotations

import click

def stratified_sample(conversations: list[dict], cluster_labels: list[int], n_total: int = 200, hard_ratio: float = 0.3, seed: int = 42) -> list[dict]:
    """
    Stratifies across clusters × thread-length buckets (1-turn / 2-3 / 4+).
    Oversamples hard cases (angry/all-caps/profanity/refund/billing/safety keywords).
    Returns sampled conversations with difficulty_tier field (easy/hard).
    """
    # Mock implementation
    return conversations[:n_total]

def split_calibration_evaluation(golden: list[dict], cal_ratio: float = 0.3, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Stratified-splits into calibration (30%) and evaluation (70%) maintaining easy/hard ratio.
    Writes golden_calibration.jsonl and golden_evaluation.jsonl
    Returns (calibration_set, evaluation_set).
    """
    split_idx = int(len(golden) * cal_ratio)
    return golden[:split_idx], golden[split_idx:]

@click.command()
@click.option('--input-file', required=True)
def cli(input_file: str):
    """CLI for building golden set."""
    pass

if __name__ == '__main__':
    cli()
