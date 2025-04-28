import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER


def validate_results_paths(result_path: Path, metrics: list, arg_name: str):
    if result_path.exists():
        df = pd.read_csv(result_path, sep='\t')
        if df.columns.tolist()[1:] != metrics:
            raise ValueError(f'`{arg_name}` already exists and the columns \
                             are different than `metrics`! Choose another \
                             path to export file or delete the existing one.')


def validate_file_extension(file_path: str, expected_extension: str) -> Path:
    path = Path(file_path)
    if path.suffix != expected_extension:
        raise argparse.ArgumentTypeError(
            f"File {file_path} does not have required extension \
            '{expected_extension}'!"
        )
    return path


def validate_args(args: argparse.Namespace):
    metrics = sorted(args.metrics)
    validate_results_paths(args.summary_results, metrics, 'summary_results')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        type=partial(validate_file_extension,
                                     expected_extension='.jsonl'),
                        required=True,
                        help='Source texts, jsonl path.')
    parser.add_argument('-r', '--reference',
                        type=partial(validate_file_extension,
                                     expected_extension='.jsonl'),
                        required=True,
                        help='Target texts (reference translations), \
                        jsonl path.')
    parser.add_argument('-H', '--hypothesis',
                        type=partial(validate_file_extension,
                                     expected_extension='.jsonl'),
                        required=True,
                        help='Hypothesis texts (candidate translations), \
                        jsonl path.')
    parser.add_argument('-m', '--metrics', nargs='+', type=str,
                        choices=['bleu','chrf','chrf2','ter','meteor','rouge'],
                        default=['bleu','chrf','chrf2','ter','meteor','rouge'])
    parser.add_argument('-f', '--full_results',
                        type=partial(validate_file_extension,
                                     expected_extension='.tsv'),
                        required=True,
                        help='Path to export TSV file of metric values \
                        for each example.')
    parser.add_argument('-S', '--summary_results',
                        type=partial(validate_file_extension,
                                     expected_extension='.tsv'),
                        required=True,
                        help='Path to export TSV file of summary of metrics \
                        results.')
    parser.add_argument('-i', '--index', type=str, required=True,
                        help='Name of the entry to be included in the summary \
                        results file.')

    return  parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)

    source = pd.read_json(args.source, orient='records', lines=True,
                          encoding='utf-8')
    reference = pd.read_json(args.reference, orient='records', lines=True,
                          encoding='utf-8')
    hypothesis = pd.read_json(args.hypothesis, orient='records', lines=True,
                              encoding='utf-8')

    assert len(source) == len(reference) == len(hypothesis), \
        "Mismatch in number of examples."

    rouge = Rouge()
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    chrf2 = CHRF(word_order=2)
    ter = TER()

    metrics = sorted(args.metrics)
    full_results = []

    values = zip(source['text'], reference['text'], hypothesis['text'])
    for src, ref, hyp in tqdm(values, total=len(source),
                              desc='Evaluating translations'):
        row = {}

        if 'bleu' in metrics:
            score = bleu.sentence_score(hyp, [ref]).score
            row['bleu'] = score

        if 'meteor' in metrics:
            score = meteor_score([ref.split()], hyp.split())
            row['meteor'] = score * 100

        if 'rouge' in metrics:
            score = rouge.get_scores(hyp, ref)[0]
            row['rouge'] = score['rouge-l']['f'] * 100

        if 'chrf' in metrics:
            score = chrf.sentence_score(hyp, [ref]).score
            row['chrf'] = score

        if 'chrf2' in metrics:
            score = chrf2.sentence_score(hyp, [ref]).score
            row['chrf2'] = score

        if 'ter' in metrics:
            score = ter.sentence_score(hyp, [ref]).score
            row['ter'] = score

        full_results.append(row)

    df_full = pd.DataFrame(full_results)
    df_full.to_csv(args.full_results, sep='\t', index=False)

    summary = {metric: df_full[metric].mean() for metric in metrics}
    df_summary = pd.DataFrame([summary], index=[args.index])
    header = not args.summary_results.exists()
    df_summary.to_csv(args.summary_results, sep='\t', header=header, mode='a+')


if __name__ == '__main__':
    main()
