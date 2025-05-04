import argparse
import time
import os
from pathlib import Path

import pandas as pd
import yaml
from groq import Groq
from tqdm import tqdm


def parse_yaml(file_path: Path) -> dict:
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    return data


def fill_placeholders(prompt: str, values: dict) -> str:
    """Replaces placeholders with their corresponding values"""
    return prompt.format(**values)


def get_answer(client: Groq, prompt: str, user_input: str,
               config: dict, model: str) -> str:

    messages = [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_input}]

    answer = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=config['temperature'],
        top_p=config['top_p'],
        stop=None,
        max_completion_tokens=config['max_completion_tokens'],
        stream=False,
    )

    return answer.choices[0].message.content.strip()


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
                        help='Size of head sample of the dataset to run.')

    return parser.parse_args()


def main():
    args = parse_args()

    system_prompt = args.system.read_text(encoding='utf-8')
    user_prompt = args.user.read_text(encoding='utf-8')
    config = parse_yaml(args.config)

    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    df = pd.read_json(args.dataset, orient='records', lines=True,
                      encoding='utf-8')

    if args.sample_size:
        df = df.head(args.sample_size)

    df['user_prompt'] = [fill_placeholders(user_prompt, record) for record in \
                         df.to_dict(orient='records')]

    output_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Generating text'):
        answer = get_answer(client, system_prompt, row['user_prompt'],
                            config, args.model)
        record = {'id': row['id'], 'text': answer}
        output_records.append(record)
        time.sleep(args.sleep) # to overcome the limit of requests per minute

    pd.DataFrame(output_records).\
        to_json(args.output, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
