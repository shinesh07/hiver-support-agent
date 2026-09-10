from __future__ import annotations

import click
import json
from datetime import datetime

@click.command()
@click.option('--model', type=click.Choice(['trivial', 'simple', 'agent']), required=True)
@click.option('--split', type=click.Choice(['calibration', 'evaluation']), required=True)
@click.option('--subsample', type=int)
@click.option('--seed', type=int, default=42)
def main(model: str, split: str, subsample: int | None, seed: int):
    """Main evaluation CLI."""
    click.echo(f"Running evaluation with model={model}, split={split}, seed={seed}")
    
    # Stub: load data, predict, run metrics
    results = {"metrics": "stub"}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/{timestamp}_{model}_{split}.json"
    
    # Save results
    with open(out_path, "w") as f:
        json.dump(results, f)
        
    click.echo(f"Results saved to {out_path}")
    click.echo("Comparison Table: Stub")

if __name__ == '__main__':
    main()
