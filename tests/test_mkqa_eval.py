import os

import tests.testdata as test_data
from mkqa_eval import (
    compute_mkqa_scores_for_language,
    MKQAAnnotation,
    MKQAPrediction,
    read_predictions,
    read_annotations,
    evaluate,
)

package_path = list(test_data.__path__)[0]


def test_compute_mkqa_scores():
    test_cases = [
        # Test 1: Perfect textual prediction
        {
            "prediction": MKQAPrediction(
                example_id="1",
                prediction="first dummy answer",
                binary_answer=None,
                no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="1", types=[], answers=["first dummy answer", "second dummy answer"],
            ),
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 2: Partially correct prediction
        {
            "prediction": MKQAPrediction(
                example_id="2",
                prediction="alternative answer",
                binary_answer=None,
                no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="2", types=[], answers=["dummy answer one", "alternative answer two"],
            ),
            "expected_f1": 0.8,
            "expected_em": 0.0,
        },
        # Test 3: Partially correct with stopword and punctuation removal
        {
            "prediction": MKQAPrediction(
                example_id="3",
                prediction="an answer ?? without stopwords",
                binary_answer=None,
                no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="3", types=[], answers=["answer -- without, stopwords!!"],
            ),
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 4: Correct No Answer prediction
        {
            "prediction": MKQAPrediction(
                example_id="4", prediction="", binary_answer=None, no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(example_id="4", types=[], answers=[""],),
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 5: Incorrect No Answer prediction
        {
            "prediction": MKQAPrediction(
                example_id="5", prediction="", binary_answer=None, no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="5", types=[], answers=["first dummy answer", "second dummy answer"],
            ),
            "expected_f1": 0.0,
            "expected_em": 0.0,
        },
        # Test 6: Incorrect No Answer prediction, when answer is binary
        {
            "prediction": MKQAPrediction(
                example_id="6", prediction="", binary_answer=None, no_answer_prob=1,
            ),
            "gold_truth": MKQAAnnotation(example_id="6", types=[], answers=["yes"],),
            "expected_f1": 0.0,
            "expected_em": 0.0,
        },
        # Test 7: Correct binary answer prediction
        {
            "prediction": MKQAPrediction(
                example_id="7", prediction="wrong answer", binary_answer="yes", no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(example_id="7", types=[], answers=["yes"],),
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 8: Incorrect binary answer prediction
        {
            "prediction": MKQAPrediction(
                example_id="8",
                prediction="distractor answer",
                binary_answer="no",
                no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(example_id="8", types=[], answers=["yes"],),
            "expected_f1": 0.0,
            "expected_em": 0.0,
        },
        # Test 9: Binary answer prediction takes precedence, but is incorrect
        {
            "prediction": MKQAPrediction(
                example_id="9", prediction="", binary_answer="no", no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(example_id="1", types=[], answers=[""],),
            "expected_f1": 0.0,
            "expected_em": 0.0,
        },
        # Test 10: No Answer probability is 1, but it is not relevant to computing initial scores in `compute_mkqa_scores_for_language`
        {
            "prediction": MKQAPrediction(
                example_id="10",
                prediction="final prediction",
                binary_answer=None,
                no_answer_prob=1.0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="10", types=[], answers=["penultimate", "final prediction"],
            ),
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
    ]

    test_preds = {ex["prediction"].example_id: ex["prediction"] for ex in test_cases}
    test_golds = {ex["gold_truth"].example_id: ex["gold_truth"] for ex in test_cases}
    expected_f1s = {ex["gold_truth"].example_id: ex["expected_f1"] for ex in test_cases}
    expected_ems = {ex["gold_truth"].example_id: ex["expected_em"] for ex in test_cases}
    test_em_scores, test_f1_scores = compute_mkqa_scores_for_language(test_preds, test_golds, "en")

    for ex_id in test_golds:
        assert (
            test_em_scores[ex_id] == expected_ems[ex_id]
        ), f"Example ID = {ex_id} | EM | Result = {test_em_scores[ex_id]} | Expected = {expected_ems[ex_id]}"
        assert (
            test_f1_scores[ex_id] == expected_f1s[ex_id]
        ), f"Example ID = {ex_id} | F1 | Result = {test_f1_scores[ex_id]} | Expected = {expected_f1s[ex_id]}"


def test_compute_mkqa_scores_in_different_languages():
    test_cases = [
        # Test 1: Test Spanish text normalization
        {
            "prediction": MKQAPrediction(
                example_id="1", prediction="esto es manzana", binary_answer=None, no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="1", types=[], answers=["esto es una manzana", "esta es otra manzana"],
            ),
            "lang": "es",
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 2: Test Arabic normalization
        {
            "prediction": MKQAPrediction(
                example_id="2", prediction="تفاحة", binary_answer=None, no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="2", types=[], answers=["التفاحة", "هذه تفاحة"],
            ),
            "lang": "ar",
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 3: Test French normalization
        {
            "prediction": MKQAPrediction(
                example_id="3",
                prediction="c'est de la pomme",
                binary_answer=None,
                no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="3", types=[], answers=["c'est la pomme", "c'est une autre pomme"],
            ),
            "lang": "fr",
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 4: Test Hungarian normalization
        {
            "prediction": MKQAPrediction(
                example_id="4", prediction="ez egy alma", binary_answer=None, no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="4", types=[], answers=["ez alma", "ez egy újabb alma"],
            ),
            "lang": "hu",
            "expected_f1": 1.0,
            "expected_em": 1.0,
        },
        # Test 5: Test Chinese Mandarin mixed segmentation f1
        {
            "prediction": MKQAPrediction(
                example_id="5", prediction="这个一个苹果", binary_answer=None, no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(example_id="5", types=[], answers=["这是还是苹果", "这是另一个苹果"],),
            "lang": "zh_cn",
            "expected_f1": 0.7692307692307692,
            "expected_em": 0,
        },
        # Test 6: Test Khmer mixed segmentation f1
        {
            "prediction": MKQAPrediction(
                example_id="6", prediction="នេះគឺជាផ្លែប៉ោម", binary_answer=None, no_answer_prob=0,
            ),
            "gold_truth": MKQAAnnotation(
                example_id="7", types=[], answers=["នេះគឺជាផ្លែប៉ោមមួយទៀត"],
            ),
            "lang": "km",
            "expected_f1": 0.8333333333333333,
            "expected_em": 0,
        },
    ]

    for case in test_cases:
        example_id = case["prediction"].example_id
        predictions = {example_id: case["prediction"]}
        gold_annotations = {example_id: case["gold_truth"]}
        lang = case["lang"]

        em_scores, f1_scores = compute_mkqa_scores_for_language(predictions, gold_annotations, lang)
        assert (
            em_scores[example_id] == case["expected_em"]
        ), f"Example ID = {example_id} | EM | Result = {em_scores[example_id]} | Expected = {case['expected_em']}"
        assert (
            f1_scores[example_id] == case["expected_f1"]
        ), f"Example ID = {example_id} | F1 | Result = {f1_scores[example_id]} | Expected = {case['expected_f1']}"


def test_compute_metrics_end_2_end():
    predictions_path = os.path.join(package_path, "en_prediction.jsonl")
    annotations_path = os.path.join(package_path, "test_mkqa.jsonl.gz")
    language = "en"
    predictions = read_predictions(predictions_path)
    annotations = read_annotations(annotations_path)

    metrics = evaluate(annotations[language], predictions, language)
    expected_metrics = {
        "best_em": 66.67,
        "best_f1": 80.95,
        "best_answerable_em": 33.33,
        "best_answerable_f1": 61.9,
        "best_unanswerable_em": 100.0,
        "best_f1_threshold": -6.91,
    }
    assert expected_metrics == dict(metrics)
