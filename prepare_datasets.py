import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


def extract_metaphorical_spans(row: pd.Series) -> list:
    tokens = row["tokens"]
    tags = row["tags"]
    spans = []
    current_span = []

    for token, tag in zip(tokens, tags):
        if tag in {1, 2}:
            current_span.append(token)
        else:
            if current_span:
                spans.append(" ".join(current_span))
                current_span = []

    if current_span:
        spans.append(" ".join(current_span))

    return spans


def include_columns(file_path):
    df = pd.read_json(file_path, orient='records', lines=True,
                      encoding='utf-8')
    df['text'] = df.tokens.str.join(' ')
    df['has_metaphor'] = df.tags.apply(lambda tags: 1 in tags)
    df['metaphorical_spans'] = df.apply(extract_metaphorical_spans, axis=1)
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)


def download_dataset():
    repo_id = 'HiTZ/meta4xnli'
    subfolder = 'detection/splits'
    local_dir = 'data/meta4xnli'

    all_files = list_repo_files(repo_id, repo_type='dataset')
    split_files = [f for f in all_files if f.startswith(subfolder + '/')]

    for file_path in split_files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type='dataset',
            local_dir=local_dir
        )
        print(f'Downloaded: {local_path}')
        include_columns(local_path)


def main():
    download_dataset()


if __name__ == '__main__':
    main()
