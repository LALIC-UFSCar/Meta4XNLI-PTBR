import argparse
from pathlib import Path

import pandas as pd


def load_datasets(source_path: Path,
                  target_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load source and target datasets from JSONL files.

    Args:
        source_path (Path): Path to the source JSONL file.
        target_path (Path): Path to the target JSONL file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Loaded source and target DataFrames.

    Raises:
        ValueError: If source and target datasets do not have the same number 
        of rows.
    """
    source_df = pd.read_json(source_path, lines=True)
    target_df = pd.read_json(target_path, lines=True)

    if len(source_df) != len(target_df):
        raise ValueError("Source and target datasets must have the same \
                         number of rows.")

    return source_df, target_df


def filter_dataset(source_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Filter the source dataset based on the 'has_metaphor' label.

    Args:
        source_df (pd.DataFrame): The source dataset.
        label (str): One of 'metaphorical', 'literal', or 'both'.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    if label == 'metaphorical':
        return source_df[source_df['has_metaphor']]
    if label == 'literal':
        return source_df[~source_df['has_metaphor']]
    return source_df


def sample_datasets(source_df: pd.DataFrame, target_df: pd.DataFrame, indices,
                    size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample the same indices from source and target datasets.

    Args:
        source_df (pd.DataFrame): Filtered source dataset.
        target_df (pd.DataFrame): Full target dataset.
        indices (pd.Index): Indices to sample from.
        size (int): Number of samples to select.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Sampled source and target datasets.

    Raises:
        ValueError: If requested sample size exceeds available filtered data.
    """
    if size > len(indices):
        raise ValueError(f"Requested sample size ({size}) exceeds available \
                         data ({len(indices)}).")

    sampled_indices = indices.to_series().sample(n=size, random_state=42).index
    sampled_indices = sorted(sampled_indices.tolist())
    return source_df.loc[sampled_indices], target_df.loc[sampled_indices]


def export_datasets(source_sample: pd.DataFrame, target_sample: pd.DataFrame,
                  source_out: Path, target_out: Path):
    """
    Export sampled datasets to JSONL files.

    Args:
        source_sample (pd.DataFrame): Sampled source dataset.
        target_sample (pd.DataFrame): Sampled target dataset.
        source_out (Path): Path to save the sampled source dataset.
        target_out (Path): Path to save the sampled target dataset.
    """
    source_sample.to_json(source_out, lines=True, orient='records',
                          force_ascii=False)
    target_sample.to_json(target_out, lines=True, orient='records',
                          force_ascii=False)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-si', '--source_input', type=Path, required=True,
                        help='Path to the source dataset.')
    parser.add_argument('-ti', '--target_input', type=Path, required=True,
                        help='Path to the target dataset.')
    parser.add_argument('-so', '--source_output', type=Path, required=True,
                        help='Path to export the source sample.')
    parser.add_argument('-to', '--target_output', type=Path, required=True,
                        help='Path to export the target sample.')
    parser.add_argument('-l', '--label', type=str, default='both',
                        choices=['metaphorical', 'literal', 'both'],
                        help='Filter by label type based on "has_metaphor" \
                            key.')
    parser.add_argument('-s', '--size', type=int, required=True,
                        help='Number of samples to draw.')
    return parser.parse_args()


def main():
    """
    Main function to run the sampling script.
    """
    args = parse_args()

    source_df, target_df = load_datasets(args.source_input, args.target_input)
    filtered_source = filter_dataset(source_df, args.label)
    source_sample, target_sample = sample_datasets(filtered_source,
                                                   target_df,
                                                   filtered_source.index,
                                                   args.size)
    export_datasets(source_sample, target_sample,
                    args.source_output, args.target_output)


if __name__ == '__main__':
    main()
