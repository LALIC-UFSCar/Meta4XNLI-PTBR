import pandas as pd


def filter_unprocessed_records(df, output_path):
    """
    Filters the DataFrame to exclude records already processed in the output file.

    Args:
        df (pd.DataFrame): The dataset loaded from args.dataset.
        output_path (Path): The path to the output file.

    Returns:
        pd.DataFrame: A filtered DataFrame with unprocessed records.
    """
    if output_path.exists():
        # Read the output file and count the number of lines
        df_processed = pd.read_json(output_path, orient='records', lines=True,
                          encoding='utf-8')

        # Filter the DataFrame to exclude already processed IDs
        df = df[~df['id'].isin(df_processed.id)]

    return df
