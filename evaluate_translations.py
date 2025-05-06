import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER

tqdm.pandas()


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
    validate_results_paths(args.summary_all, metrics, 'summary_all')
    validate_results_paths(args.summary_metaphors, metrics,'summary_metaphors')
    validate_results_paths(args.summary_literals, metrics, 'summary_literals')


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
                        choices=['bleu','chrf','chrf2','meteor','rouge','ter'],
                        default=['bleu','chrf','chrf2','meteor','rouge','ter'])
    parser.add_argument('-f', '--full_results',
                        type=partial(validate_file_extension,
                                     expected_extension='.tsv'),
                        required=True,
                        help='Path to export TSV file of metric values \
                        for each example.')
    parser.add_argument('-a', '--summary_all',
                        type=partial(validate_file_extension,
                                     expected_extension='.tsv'),
                        help='Path to export TSV file of summary of metrics \
                        for metaphorical and literal examples.')
    parser.add_argument('-M', '--summary_metaphors',
                        type=partial(validate_file_extension,
                                     expected_extension='.tsv'),
                        help='Path to export TSV file of summary of metrics \
                        for metaphorical examples.')
    parser.add_argument('-l', '--summary_literals',
                        type=partial(validate_file_extension,
                                     expected_extension='.tsv'),
                        help='Path to export TSV file of summary of metrics \
                        for literal examples.')
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

    df = pd.merge(source.rename(columns={'text': 'src'})\
                    [['id','src','has_metaphor']],
                  reference.rename(columns={'text': 'ref'})[['id','ref']],
                  how='inner', on='id')
    df = pd.merge(df,
                  hypothesis.rename(columns={'text': 'hyp'})[['id','hyp']],
                  how='inner', on='id')

    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    chrf2 = CHRF(word_order=2)
    rouge = Rouge()
    ter = TER()

    if 'bleu' in args.metrics:
        df['bleu'] = df.progress_apply(lambda x: \
                                      bleu.sentence_score(x['hyp'],
                                                          [x['ref']]).score,
                                      axis=1)

    if 'chrf' in args.metrics:
        df['chrf'] = df.progress_apply(lambda x: \
                                       chrf.sentence_score(x['hyp'],
                                                           [x['ref']]).score,
                                       axis=1)

    if 'chrf2' in args.metrics:
        df['chrf2'] = df.progress_apply(lambda x: \
                                        chrf2.sentence_score(x['hyp'],
                                                            [x['ref']]).score,
                                        axis=1)

    if 'meteor' in args.metrics:
        df['meteor'] = df.progress_apply(lambda x: \
                                         meteor_score([x['ref'].split()],
                                                      x['hyp'].split()) *100,
                                         axis=1)

    if 'rouge' in args.metrics:
        df['rouge'] = df.progress_apply(lambda x: rouge.get_scores(
                                            x['hyp'],
                                            x['ref'])[0]['rouge-l']['f']*100,
                                        axis=1)

    if 'ter' in args.metrics:
        df['ter'] = df.progress_apply(lambda x: \
                                      ter.sentence_score(x['hyp'],
                                                         [x['ref']]).score,
                                      axis=1)

    df.drop(columns=['src','ref','hyp']).\
        to_csv(args.full_results, sep='\t', index=False)

    if args.summary_all is not None:
        df_summary_all = df[args.metrics].mean().to_frame(name=args.index).T
        header = not args.summary_all.exists()
        df_summary_all.to_csv(args.summary_all, sep='\t',
                              header=header, mode='a+')

    if args.summary_metaphors is not None:
        df_summary_metaphors = df[df.has_metaphor][args.metrics]\
                                .mean().to_frame(name=args.index).T
        header = not args.summary_metaphors.exists()
        df_summary_metaphors.to_csv(args.summary_metaphors, sep='\t',
                                    header=header, mode='a+')

    if args.summary_literals is not None:
        df_summary_literals = df[~df.has_metaphor][args.metrics]\
                                    .mean().to_frame(name=args.index).T
        header = not args.summary_literals.exists()
        df_summary_literals.to_csv(args.summary_literals, sep='\t',
                                header=header, mode='a+')


if __name__ == '__main__':
    main()
