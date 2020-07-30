import argparse
import logging
import os
import sys
from collections import defaultdict
from multiprocessing.pool import Pool
from typing import Optional, Dict

import pandas as pd
from tabulate import tabulate

import mkqa_eval
import mkqa_eval_util as eval_util


def parse_args():
    parser = argparse.ArgumentParser("Official evaluation script for all MKQA languages.")
    parser.add_argument(
        "-a",
        "--annotation_file",
        metavar="mkqa.jsonl.gz",
        required=True,
        help="Input annotations MKQA JSON Lines gzip file.",
    )
    parser.add_argument(
        "-p",
        "--predictions_dir",
        metavar="preds/",
        required=True,
        help="Model predictions for each language",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        metavar="results/",
        help="Write evaluation metrics to files (default is stdout).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def read_prediction_dir(predictions_dir: str) -> Dict[str, Dict[str, mkqa_eval.MKQAAnnotation]]:
    """Read a directory that contains predictions for all languages

    Args:
        predictions_dir:  a directory that contains predictions for all languages

    Returns:
        A mapping from example id to MKQAPrediction for all languages
    """

    assert os.path.exists(predictions_dir)
    all_language_predictions = {}
    for language in mkqa_eval.MKQA_LANGUAGES:
        prediction_file = os.path.join(predictions_dir, f"{language}.jsonl")
        if not os.path.exists(prediction_file):
            logging.info(
                f"WARNING: Missing predictions file for language `{language}`. Expecting file `{prediction_file}``"
            )
            continue

        all_language_predictions[language] = mkqa_eval.read_predictions(prediction_file)

    return all_language_predictions


def evaluate_predictions(
    annotations: Dict[str, mkqa_eval.MKQAAnnotation],
    predictions: Dict[str, mkqa_eval.MKQAPrediction],
    language: str,
    out_dir: str = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate predictions for one language from its prediction file and MKQAAnnotations

    Args:
        annotations: a mapping from example id to corresponding MKQAAnnotation
        predictions: a mapping from example id to corresponding MKQAPrediction
        language: language code in MKQA_LANGUAGES
        out_dir: (Optional) Saves evaluation results into this directory.
        verbose: (Optional) Collection additional metrics

    Returns:
        A dictionary of metrics for the specified language
    """
    metrics_by_language = {}

    metrics = mkqa_eval.evaluate(
        annotations=annotations,
        predictions=predictions,
        language=language,
        out_dir=os.path.join(out_dir, f"{language}_results/") if out_dir else None,
        verbose=verbose,
        print_metrics=False,
    )
    metrics_by_language[language] = metrics
    return metrics_by_language


def evaluate_all_languages(
    annotations_path: str,
    all_language_predictions: Dict[str, Dict[str, mkqa_eval.MKQAAnnotation]],
    out_dir: Optional[str] = None,
    verbose: bool = False,
    serialize: bool = False,
):
    """Evaluate all predictions for all supported languages

    Args:
        annotations_path: annotations MKQA JSON Lines gzip file.
        all_language_predictions: a mapping from example id to MKQAPrediction for all languages
        out_dir: (Optional) saves evaluation results into this directory.
        verbose: (Optional) print out additional metrics
        serialize: (Optional) disable parallel processing

    """
    assert os.path.exists(annotations_path), "Missing MKQA annotation file"

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    all_annotations = mkqa_eval.read_annotations(annotations_path)

    metrics_by_language = {}
    if not serialize:
        with Pool() as pool:
            values = pool.starmap(
                evaluate_predictions,
                [
                    (
                        all_annotations[language],
                        all_language_predictions[language],
                        language,
                        out_dir,
                        verbose,
                    )
                    for language in all_language_predictions.keys()
                ],
            )
    else:
        values = []
        for language in all_language_predictions.keys():
            values.append(
                evaluate_predictions(
                    all_annotations[language],
                    all_language_predictions[language],
                    language,
                    out_dir,
                    verbose,
                )
            )

    for value in values:
        metrics_by_language.update(value)

    # Here we compute the macro average over the languages for which prediction files were supplied for.
    # The official macro-average requires all 26 prediction files to be included.
    metrics_by_language["Macro Average"] = eval_util.aggregate_summaries(
        list(metrics_by_language.values())
    )

    logging.info("Metrics by languages:")
    metrics_summary = defaultdict(list)
    for language, metrics in metrics_by_language.items():
        metrics_summary["language"].append(language)
        for key, value in metrics.items():
            metrics_summary[key].append(value)

    metrics_df = pd.DataFrame(metrics_summary)
    print(tabulate(metrics_df, headers="keys", tablefmt="psql", showindex=False))

    if out_dir:
        metrics_df.to_json(os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    args = parse_args()
    all_language_predictions = read_prediction_dir(args.predictions_dir)
    evaluate_all_languages(
        args.annotation_file, all_language_predictions, args.out_dir, args.verbose,
    )
