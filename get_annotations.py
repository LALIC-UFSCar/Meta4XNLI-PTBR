import argparse
import ast
import time
import os
from pathlib import Path

import jsonlines
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from tqdm import tqdm

from utils.io import parse_yaml
from utils.llm_request import fill_placeholders, get_answer


def get_prompt(prompt: str, row: pd.Series) -> str:
    return fill_placeholders(prompt, row.to_dict())


def str_to_list(input_str: str) -> list:
    """
    Converts a string that looks like a Python list into an actual list.
    
    Args:
        input_str (str): The input string representing a list.
        
    Returns:
        list: The converted list if the input is valid.
        None: If the input is invalid.
    """
    try:
        result = ast.literal_eval(input_str)
        return result
    except (ValueError, SyntaxError) as e:
        print(f"Error: Invalid input string. {e}")
        return None


def is_annotation_valid(annotation: list, tokens: list) -> bool:
    if len(annotation) != len(tokens):
        return False
    return all(el in {0, 1, 2} for el in annotation)


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=Path, required=True,
                        help='Path to dataset in .jsonl format.')
    parser.add_argument('-s', '--system', type=Path, required=True,
                        help='Path to system prompt file.')
    parser.add_argument('-u', '--user', type=Path, required=True,
                        help='Path to user prompt file. It can contain \
                        placeholders, indicated between braces.')
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help='Request configurations related to model, \
                        temperature, etc.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Path to store the generations.')
    parser.add_argument('-z', '--sleep', type=int, required=False, default=2,
                        help='Seconds parameter of time.sleep.')
    parser.add_argument('-S', '--sample_size', type=int, required=False,
                        help='Size of head sample of the dataset to run. \
                            Mostly used for debugging.')
    parser.add_argument('-C', '--client', type=str, choices=['openai', 'groq'],
                        help='Define the client.', required=True)
    parser.add_argument('-l', '--limit', type=int, required=False, default=1,
                        help='Limit of times to try the request.Default to 0.')

    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv()

    system_prompt = args.system.read_text(encoding='utf-8')
    user_prompt = args.user.read_text(encoding='utf-8')
    config = parse_yaml(args.config)

    if args.client == 'groq':
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    else:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    df = pd.read_json(args.dataset, orient='records', lines=True,
                      encoding='utf-8')

    df = filter_unprocessed_records(df, args.output)

    if args.sample_size and len(df) > args.sample_size:
        df = df.head(args.sample_size)

    df['user_prompt'] = [fill_placeholders(user_prompt, record) for record in \
                         df.to_dict(orient='records')]

    with open(args.output, 'a+', encoding='utf-8') as file:
        with jsonlines.Writer(file) as writer:
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc='Generating text'):
                system_prompt = get_prompt(system_prompt, row)

                attempts = 0
                while attempts < args.limit:
                    answer = get_answer(client, system_prompt,
                                        row['user_prompt'], config, args.model)
                    answer = str_to_list(answer)
                    # to overcome the limit of requests per minute
                    time.sleep(args.sleep)
                    attempts += 1

                    if isinstance(answer, list) and \
                        is_annotation_valid(answer, row['tokens']):
                        record = {'id': row['id'], 'annotation': answer}
                        writer.write(record)
                        break

                    print(f'Id {row["id"]}: invalid answer: {answer}. ' + \
                          f'Has {len(answer)} elements, ' + \
                          f'should have {len(row["tokens"])}. Retrying...')


if __name__ == '__main__':
    main()
