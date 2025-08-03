import argparse
import time
import os
import re
import sys
from pathlib import Path

import jsonlines
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from tqdm import tqdm

from utils.data_processing import filter_unprocessed_records
from utils.io import parse_yaml
from utils.llm_request import get_answer, fill_placeholders


def spans_to_text(spans: list) -> str:
    """Converts a list of text spans into a single formatted string."""
    pattern = re.compile(r'[,.!?]$')
    spans = [pattern.sub('', span) for span in spans]

    if len(spans) == 0:
        return ''
    if len(spans) == 1:
        return f'"{spans[0]}"'
    if len(spans) == 2:
        return f'"{spans[0]}" and "{spans[1]}"'
    return ', '.join(f'"{span}"' for span in spans[:-1])+ f' and "{spans[-1]}"'


def get_prompt(main_prompt: str, extra_prompt: str, row: pd.Series) -> str:
    main_prompt = fill_placeholders(main_prompt, row.to_dict())
    if extra_prompt is None:
        return main_prompt
    extra_prompt = fill_placeholders(extra_prompt, row.to_dict())
    if row['has_metaphor']:
        return main_prompt
    return extra_prompt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=Path, required=True,
                        help='Path to dataset in .jsonl format.')
    parser.add_argument('-s', '--system', type=Path, required=True,
                        help='Path to system prompt file. If \
                        `additional_system` is passed, this parameter \
                        represents the system prompt for METAPHORICAL \
                        examples. Otherwise, it represents the system prompt \
                        for all examples.')
    parser.add_argument('-a', '--additional_system', type=Path, required=False,
                        help='If passed, represents the system prompt for \
                        LITERAL examples.')
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
                        help='Size of head sample of the dataset to run.')
    parser.add_argument('-C', '--client', type=str, choices=['openai', 'groq'],
                        help='Define the client.', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv()

    main_system_prompt = args.system.read_text(encoding='utf-8')
    if args.additional_system is not None:
        extra_system_prompt=args.additional_system.read_text(encoding='utf-8')
    else:
        extra_system_prompt = None
    user_prompt = args.user.read_text(encoding='utf-8')
    config = parse_yaml(args.config)

    if args.client == 'groq':
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    else:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    df = pd.read_json(args.dataset, orient='records', lines=True,
                      encoding='utf-8')

    df = filter_unprocessed_records(df, args.output)

    if len(df) == 0:
        print('No unprocessed records found in the dataset.')
        sys.exit()

    if args.sample_size and len(df) > args.sample_size:
        df = df.head(args.sample_size)

    df['user_prompt'] = [fill_placeholders(user_prompt, record) for record in \
                         df.to_dict(orient='records')]

    if '{metaphorical_spans_text}' in main_system_prompt \
        or '{metaphorical_spans_text}' in extra_system_prompt:
        df['metaphorical_spans_text'] = df.metaphorical_spans.\
            apply(spans_to_text)

    file = open(args.output, 'a+')
    with jsonlines.Writer(file) as writer:
        for _, row in tqdm(df.iterrows(),total=len(df),desc='Generating text'):
            system_prompt = get_prompt(main_system_prompt,extra_system_prompt,row)
            answer = get_answer(client, system_prompt, row['user_prompt'],
                                config, args.model)
            record = {'id': row['id'], 'text': answer}
            writer.write(record)
            time.sleep(args.sleep) # to overcome the limit of requests per minute

    writer.close()


if __name__ == '__main__':
    main()
