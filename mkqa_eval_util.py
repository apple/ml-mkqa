import collections
import os
import re
import string
from collections import Counter, OrderedDict
from multiprocessing import Pool
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

MIXED_SEGMENTATION_LANGS = ["zh_cn", "zh_hk", "zh_tw", "ja", "th", "km"]

ARTICLE_REGEX_BY_LANG = {
    "en": r"\b(a|an|the)\b",
    "es": r"\b(un|una|unos|unas|el|la|los|las)\b",
    "vi": r"\b(của|là|cái|chiếc|những)\b",
    "de": r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
    "ar": "\sال^|ال",
    "nl": r"\b(de|het|een|des|der|den)\b",
    "sv": r"\b(en|ett)\b",
    "da": r"\b(en|et)\b",
    "no": r"\b(en|et|ei)\b",
    "fr": r"\b(le|la|l'|les|du|de|d'|des|un|une|des)",
    "pt": r"\b(o|a|os|as|um|uma|uns|umas)\b",
    "it": r"\b(il|lo|la|l'|i|gli|le|del|dello|della|dell'|dei|degli|degl'|delle|un'|uno|una|un)",
    "fi": r"\b(se|yks|yksi)\b",
    "hu": r"\b(a|az|egy)\b",
}


def map_em_value(prediction, gold_answers, lang):
    em_value = compute_max_score_over_answers(calculate_em, prediction, gold_answers, lang)
    return float(em_value)


def map_f1_value(prediction, gold_answers, lang):
    f1_value = compute_max_score_over_answers(calculate_f1, prediction, gold_answers, lang)
    return float(f1_value)


def get_text_metrics(
    predictions: List[str], gold_answers: List[List[str]], lang: str, serial=True, workers=None
) -> Dict[str, List[float]]:
    """Compute metrics from the predicted and answer texts."""
    if serial:
        f1_scores = [
            map_f1_value(predictions[i], gold_answers[i], lang) for i in range(len(predictions))
        ]
        em_scores = [
            map_em_value(predictions[i], gold_answers[i], lang) for i in range(len(predictions))
        ]
    else:
        with Pool(workers) as p:
            f1_scores = p.starmap(
                map_f1_value,
                [(predictions[i], gold_answers[i], lang) for i in range(len(predictions))],
                chunksize=64,
            )
            em_scores = p.starmap(
                map_em_value,
                [(predictions[i], gold_answers[i], lang) for i in range(len(predictions))],
                chunksize=64,
            )

    return {"f1": f1_scores, "exact_match": em_scores}


def summarize_default_metrics(
    em_scores, f1_scores, qid_is_answerable, metrics: Optional[Dict[str, float]] = None,
):
    """Summarize EM and F1 based on default threshold"""
    assert set(em_scores.keys()) == set(f1_scores.keys()) == set(qid_is_answerable.keys())
    ans_em_scores = {qid: em_scores[qid] for qid in em_scores if qid_is_answerable[qid]}
    ans_f1_scores = {qid: f1_scores[qid] for qid in f1_scores if qid_is_answerable[qid]}
    unans_em_scores = {qid: em_scores[qid] for qid in em_scores if not qid_is_answerable[qid]}

    summary = OrderedDict(
        [
            ("exact_match", round(100.0 * np.mean(list(em_scores.values())), 2)),
            ("f1", round(100.0 * np.mean(list(f1_scores.values())), 2)),
            ("answerable_exact_match", round(100.0 * np.mean(list(ans_em_scores.values())), 2)),
            ("answerable_f1", round(100.0 * np.mean(list(ans_f1_scores.values())), 2)),
            ("unanswerable_exact_match", round(100.0 * np.mean(list(unans_em_scores.values())), 2)),
        ]
    )

    if metrics:
        metrics.update(summary)
    return summary


def aggregate_summaries(dicts):
    summaries = collections.defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            assert isinstance(v, float) or isinstance(v, int)
            summaries[k].append(v)
    results = {}
    for k, v in summaries.items():
        results[k] = round(float(np.mean(v)), 2)

    return results


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if temp_str != "":
            ss = whitespace_tokenize(temp_str)
            segs_out.extend(ss)
            temp_str = ""
        segs_out.append(char)

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer_by_language(s, lang):
    """Lower text, remove punctuation, articles and extra whitespace.
    This function is customized by language.
    """

    def remove_articles(text, lang):
        article_regex = ARTICLE_REGEX_BY_LANG.get(lang)
        if article_regex:
            return re.sub(article_regex, " ", text)
        else:
            return text

    def white_space_fix(text, lang):

        if lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


def plot_f1(answerable_f1_by_id, unanswerable_em_by_id, na_probs_by_id, qid_to_has_ans, image_dir):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    qid_list = sorted(na_probs_by_id, key=lambda k: na_probs_by_id[k])
    question_counts = len(qid_list)
    answerable_f1 = []
    overall_f1 = []
    unanswerable_em = []
    thresholds = []

    sum_answerable_f1 = 0
    sum_unanswerable_em = num_no_ans
    for i, qid in enumerate(qid_list):
        thresholds.append(na_probs_by_id[qid])
        if qid in answerable_f1_by_id:
            sum_answerable_f1 += answerable_f1_by_id[qid]
        elif qid in unanswerable_em_by_id:
            sum_unanswerable_em += unanswerable_em_by_id[qid] - 1
        else:
            raise ValueError(f"{qid} is not in either answerable or unanswerable predictions")

        answerable_f1.append(sum_answerable_f1 / question_counts)
        unanswerable_em.append(sum_unanswerable_em / question_counts)
        overall_f1.append((sum_answerable_f1 + sum_unanswerable_em) / question_counts)

    plt.plot(thresholds, answerable_f1, color="green", label="Answerable F1")
    plt.plot(thresholds, unanswerable_em, color="red", label="Unanswerable F1")
    plt.plot(thresholds, overall_f1, color="blue", label="Overall F1")
    plt.legend()

    plt.xlabel("No Answer Threshold")
    plt.ylabel("F1")
    plt.title("F1 plot for different answer types")
    plt.savefig(os.path.join(image_dir, "f1_plot.png"))
    plt.clf()


def calculate_em(prediction, gold_answer, language):
    norm_pred = normalize_answer_by_language(prediction, language)
    norm_answer = normalize_answer_by_language(gold_answer, language)
    return int(norm_pred == norm_answer)


def calculate_f1(prediction, gold_answer, language):
    gold_toks = normalize_answer_by_language(gold_answer, language).split() if gold_answer else []
    pred_toks = normalize_answer_by_language(prediction, language).split() if prediction else []
    common = Counter(gold_toks) & Counter(pred_toks)
    num_common = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If the prediction or gold_answer is No Answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_common == 0:
        return 0.0

    recall = 1.0 * num_common / len(gold_toks)
    precision = 1.0 * num_common / len(pred_toks)
    return (2.0 * precision * recall) / (precision + recall)


def compute_max_score_over_answers(metric_fn, prediction, ground_truths, language):
    assert len(ground_truths) > 0, "Gold truth answers list should never be empty."
    scores_by_answer = [
        metric_fn(prediction, ground_truth, language) for ground_truth in ground_truths
    ]
    return max(scores_by_answer)


def compute_best_score_and_threshold(
    predictions, scores, no_answer_probs, qid_has_answer
) -> Dict[str, float]:
    # Begin at threshold of 0, where all predictions are No Answer.
    best_threshold = 0.0
    current_score = best_score = sum(1 for k in qid_has_answer if not qid_has_answer[k])

    exs_sorted_by_na_prob = sorted(no_answer_probs, key=lambda k: no_answer_probs[k])
    for qid in exs_sorted_by_na_prob:

        if qid_has_answer[qid]:  # Gold truth is answer, and we predict an answer
            score_diff = scores[qid]
        elif predictions[qid]:  # If gold truth is No Answer, but we predict an answer
            score_diff = -1
        else:  # If gold truth and prediction are both No Answer
            score_diff = 0
        current_score += score_diff

        # Update best score and threshold if new max value
        if current_score > best_score:
            best_threshold = no_answer_probs[qid]
            best_score = current_score

    return {
        "best_score": 100.0 * best_score / len(scores),
        "best_threshold": best_threshold,
    }


def apply_no_answer_threshold(scores, no_answer_probs, qid_has_answer, no_answer_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_no_answer = no_answer_probs[qid] > no_answer_thresh
        new_scores[qid] = float(not qid_has_answer[qid]) if pred_no_answer else s
    return new_scores


def plot_na_prob_histogram(no_answer_probs, qid_list, outdir, name):
    x = [no_answer_probs[k] for k in qid_list]
    weights = np.ones_like(x) / float(len(x))
    plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
    plt.xlabel("No Answer Probability")
    plt.ylabel("Proportion of Dataset")
    plt.title(f"No Answer Probability Histogram: {name}")
    plt.savefig(os.path.join(outdir, f"na_prob_histogram_{name}.png"))
    plt.clf()
