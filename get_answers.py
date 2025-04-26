import argparse
import time
import os
from itertools import repeat
from pathlib import Path

import pandas as pd
from groq import Groq
from tqdm import tqdm


def fill_placeholders(prompt: str, values: dict) -> str:
    """Replaces placeholders with their corresponding values"""
    return prompt.format(**values)


def get_answer(prompt: str, user_input: str) -> str:
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    messages = [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_input}]

    answer = client.chat.completions.create(
        messages=messages,
        model='llama-3.1-8b-instant',
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    return answer.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=Path, required=True,
                        help='Path to dataset in .jsonl format.')
    parser.add_argument('-s', '--system', type=Path, required=True,
                        help='Path to system prompt file.')
    parser.add_argument('-u', '--user', type=Path, required=True,
                        help='Path to user prompt file. It can contain \
                        placeholders, indicated between braces.')
    parser.add_argument('-o', '--output_path', type=Path, required=True,
                        help='Path to store the generations.')
    parser.add_argument('-z', '--sleep', type=int, required=False, default=2,
                        help='Seconds parameter of time.sleep.')
    args = parser.parse_args()

    with open(args.system, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    with open(args.user, "r", encoding="utf-8") as f:
        user_prompt = f.read()

    df = pd.read_json(args.dataset, orient='records', lines=True,
                      encoding='utf-8')

    df = df.head(5)

    records = df.to_dict(orient='records')
    user_prompts = map(fill_placeholders, repeat(user_prompt), records)
    user_prompts = list(user_prompts)
    df['user_prompt'] = user_prompts

    output_records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        answer = get_answer(system_prompt, row['user_prompt'])
        record = {'id': row['id'], 'text': answer}
        output_records.append(record)
        time.sleep(args.sleep) # to overcome the limit of requests per minute

    df_out = pd.DataFrame(output_records)
    df_out.to_json(args.output_path, orient='records', lines=True,
                   force_ascii=False)


if __name__ == '__main__':
    main()
