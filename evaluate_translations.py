import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from evaluate import load
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER

from utils.args_validation import validate_file_extension

tqdm.pandas()
torch.set_float32_matmul_precision('high')


def validate_results_paths(result_path: Path, metrics: list, arg_name: str):
    if result_path is not None and result_path.exists():
        df = pd.read_csv(result_path, sep='\t')
        if df.columns.tolist()[1:] != metrics:
            print(f'Columns of the existing file: {df.columns.tolist()[1:]}')
            print(f'Metrics to be generated: {metrics}')
            raise ValueError(f'`{arg_name}` already exists and the columns ' \
                             'are different than `metrics`! Choose another ' \
                             'path to export file or delete the existing one.')


def validate_args(args: argparse.Namespace):
    validate_results_paths(args.summary_all, args.metrics, 'summary_all')
    validate_results_paths(args.summary_metaphors, args.metrics,'summary_metaphors')
    validate_results_paths(args.summary_literals, args.metrics, 'summary_literals')


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
                        choices=['bertscore_f1','bertscore_precision',
                                 'bertscore_recall','bleu','chrf','chrf2',
                                 'comet_no_ref','comet_ref',
                                 'meteor','rouge','ter'],
                        default=['bertscore_f1','bertscore_precision',
                                 'bertscore_recall','bleu','chrf','chrf2',
                                 'comet_no_ref','comet_ref',
                                 'meteor','rouge','ter'],)
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

    bertscore_metrics =  {'bertscore_f1', 'bertscore_precision',
                          'bertscore_recall'}
    if bertscore_metrics.intersection(set(args.metrics)):
        bertscore = load("bertscore")
        scores = bertscore.compute(predictions=df['hyp'].values.tolist(),
                                   references=df['ref'].values.tolist(),
                                   model_type="distilbert-base-uncased")
        df['bertscore_f1'] = scores['f1']
        df['bertscore_precision'] = scores['precision']
        df['bertscore_recall'] = scores['recall']

    if 'bleu' in args.metrics:
        bleu = BLEU(effective_order=True)
        df['bleu'] = df.progress_apply(lambda x: \
                                      bleu.sentence_score(x['hyp'],
                                                          [x['ref']]).score,
                                      axis=1)

    if 'chrf' in args.metrics:
        chrf = CHRF()
        df['chrf'] = df.progress_apply(lambda x: \
                                       chrf.sentence_score(x['hyp'],
                                                           [x['ref']]).score,
                                       axis=1)

    if 'chrf2' in args.metrics:
        chrf2 = CHRF(word_order=2)
        df['chrf2'] = df.progress_apply(lambda x: \
                                        chrf2.sentence_score(x['hyp'],
                                                            [x['ref']]).score,
                                        axis=1)

    if 'comet_no_ref' in args.metrics:
        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)
        records = df.rename(columns={'hyp':'mt'})[['src','mt']].\
                    to_dict('records')
        scores = model.predict(records, batch_size=8, gpus=1).scores
        df['comet_no_ref'] = scores

    if 'comet_ref' in args.metrics:
        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)
        records = df.rename(columns={'hyp':'mt'})[['src','mt','ref']].\
                    to_dict('records')
        scores = model.predict(records, batch_size=8, gpus=1).scores
        df['comet_ref'] = scores

    if 'meteor' in args.metrics:
        df['meteor'] = df.progress_apply(lambda x: \
                                         meteor_score([x['ref'].split()],
                                                      x['hyp'].split()) *100,
                                         axis=1)

    if 'rouge' in args.metrics:
        rouge = Rouge()
        df['rouge'] = df.progress_apply(lambda x: rouge.get_scores(
                                            x['hyp'],
                                            x['ref'])[0]['rouge-l']['f']*100,
                                        axis=1)

    if 'ter' in args.metrics:
        ter = TER()
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
