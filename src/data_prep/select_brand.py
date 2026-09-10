from __future__ import annotations

import click
import pandas as pd

def compute_brand_stats(raw_data_path: str) -> pd.DataFrame:
    """
    Computes per-brand statistics:
    - lexical_diversity: unique trigram ratio
    - self_similarity: mean pairwise embedding cosine sim of ~200 random outbound replies
    - reply_length_distribution: median + IQR of token counts
    - resolution_proxy_rate: rate of successful resolution
    """
    # Placeholder implementation
    return pd.DataFrame()

def rank_brands(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Computes composite score = (lexical_diversity * resolution_proxy_rate) / self_similarity
    Ranks brands based on this score.
    """
    if not stats.empty:
        stats['composite_score'] = (stats['lexical_diversity'] * stats['resolution_proxy_rate']) / stats['self_similarity']
        return stats.sort_values('composite_score', ascending=False)
    return stats

@click.command()
@click.option('--raw-data', required=True, help="Path to raw data file")
@click.option('--output', default="ranked_brands.csv", help="Output path")
def cli(raw_data: str, output: str):
    """CLI entry point for selecting a brand."""
    stats = compute_brand_stats(raw_data)
    ranked = rank_brands(stats)
    ranked.to_csv(output, index=False)
    click.echo(f"Saved ranked brands to {output}")

if __name__ == '__main__':
    cli()
