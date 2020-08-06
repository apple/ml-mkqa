# MKQA: Multilingual Knowledge Questions & Answers

[**Tasks**](#task-description) | [**Dataset**](#dataset) | [**Leaderboard**](#leaderboard) | [**Evaluation**](#evaluation) |
[**Paper**](https://arxiv.org/abs/2007.15207) |
[**Citation**](#citation) | [**License**](#license)

We introduce MKQA, an open-domain question answering evaluation set comprising 10k question-answer pairs 
aligned across 26 typologically diverse languages (260k question-answer pairs in total).
The goal of this dataset is to provide a challenging benchmark for question answering quality across a wide set of
languages. Please refer to our paper for details, [MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering](https://arxiv.org/abs/2007.15207)


## Task Description
Given a question <code>q<sup>l</sup></code> in language `l`, the task is to produce a prediction <code>p<sup>l</sup></code> in {No Answer, Yes, No, Text Answer}, 
where a Text Answer is a span of tokens in the corresponding language. 
<code>p<sup>l</sup></code> can be obtained by any method, extracted from a document, generated, or derived from a knowledge graph.
Wherever possible, textual answers are accompanied by Wikidata QIDs, for entity linking and evaluating knowledge graph approaches. 
These QIDs also enable automatic translations for most answers into any Wikipedia language through the Wikidata knowledge graph.


## Dataset
MKQA contains 10,000 queries sampled from the [Google Natural Questions dataset](https://github.com/google-research-datasets/natural-questions).  

For each query we collect new passage-independent answers. 
These queries and answers are then human translated into 25 Non-English languages.
MKQA data can be downloaded from [here](dataset/mkqa.jsonl.gz).

Each example in the dataset contains the unique Natural Questions `example_id`, the original English `query`, and then `queries` and `answers` in 26 languages.

```
{
 'example_id': 563260143484355911,
 'queries': {
  'en': "who sings i hear you knocking but you can't come in",
  'ru': "кто поет i hear you knocking but you can't come in",
  'ja': '「 I hear you knocking」は誰が歌っていますか',
  'zh_cn': "《i hear you knocking but you can't come in》是谁演唱的",
  ...
 },
 'query': "who sings i hear you knocking but you can't come in",
 'answers': {'en': [{'type': 'entity',
    'entity': 'Q545186',
    'text': 'Dave Edmunds',
    'aliases': []}],
  'ru': [{'type': 'entity',
    'entity': 'Q545186',
    'text': 'Эдмундс, Дэйв',
    'aliases': ['Эдмундс', 'Дэйв Эдмундс', 'Эдмундс Дэйв', 'Dave Edmunds']}],
  'ja': [{'type': 'entity',
    'entity': 'Q545186',
    'text': 'デイヴ・エドモンズ',
    'aliases': ['デーブ・エドモンズ', 'デイブ・エドモンズ']}],
  'zh_cn': [{'type': 'entity', 'text': '戴维·埃德蒙兹 ', 'entity': 'Q545186'}],
  ...
  },
}
```
Each answer is labelled with an answer type. The breakdown is:

| Answer Type | Occurrence |
|---------------|---------------|
| `entity`               | `4221`             |
| `long_answer`          | `1815`             |
| `unanswerable`         | `1427`             |
| `date`                 | `1174`             |
| `number`               | `485`              |
| `number_with_unit`     | `394`              |
| `short_phrase`         | `346`              |
| `binary`               | `138`              |
  
For each language, there can be more than one acceptable textual answer, in order to capture a variety of possible valid answers. 
All the supported languages are:  

| Language code | Language name |
|---------------|---------------|
| `ar`     | `Arabic`                    |
| `da`     | `Danish`                    |
| `de`     | `German`                    |
| `en`     | `English`                   |
| `es`     | `Spanish`                   |
| `fi`     | `Finnish`                   |
| `fr`     | `French`                    |
| `he`     | `Hebrew`                    |
| `hu`     | `Hungarian`                 |
| `it`     | `Italian`                   |
| `ja`     | `Japanese`                  |
| `ko`     | `Korean`                    |
| `km`     | `Khmer`                    |
| `ms`     | `Malay`                     |
| `nl`     | `Dutch`                     |
| `no`     | `Norwegian`                 |
| `pl`     | `Polish`                    |
| `pt`     | `Portuguese`                |
| `ru`     | `Russian`                   |
| `sv`     | `Swedish`                   |
| `th`     | `Thai`                      |
| `tr`     | `Turkish`                   |
| `vi`     | `Vietnamese`                |
| `zh_cn`     | `Chinese (Simplified)`   |
| `zh_hk`     | `Chinese (Hong kong)`    |
| `zh_tw`     | `Chinese (Traditional)`  |


## Leaderboard
| Model name | Best Overall F1 | Best Answerable F1 | Best Unnswerable F1 | Link to paper |
|---------------|---------------|---------------|---------------|---------------|
| (Baseline) XLM-R Large Translate Train   |46.0        | 27.6        |84.5        |https://arxiv.org/pdf/1911.02116.pdf        |

> Submit a pull request to this repository to add yourself to the MKQA leaderboard. Scores are ordered by Best Overall F1.

## Evaluation
The official evaluation scripts provide two ways to evaluate performance on the MKQA dataset

The evaluation script expects a json lines (jsonl) prediction file with a specific format:
```
{
  "example_id": -7449157003522518870,
  "prediction": "Hafþór Júlíus `` Thor '' Björnsson",
  "binary_answer": null,
  "no_answer_prob": 0.23618
}
...
```

### Evaluate performance for single language 
To evaluate prediscion for a single language, use `mkqa_eval.py` script and indicate prediction language
```
python mkqa_eval.py --annotation_file ./dataset/mkqa.jsonl.gz \
 --predictions_file ./sample_predictions/en.jsonl \
 --language en \
 (--out_dir <optional output directory for saving metrics and pr curves>, --verbose) 
```

### Evaluate performances for all languages
To evaluate predictions for all languages, use the following script. Save the prediction file for each language in the same directory and name each prediction file by its language code, such as `en.jsonl`
```
python mkqa_eval_all_languages.py --annotation_file ./dataset/mkqa.jsonl.gz \
 --predictions_dir ./sample_predictions \
 (--out_dir <optional output directory for saving metrics and pr curves>, --verbose)
``` 

To get our the zero shot multilingual bert baseline, use the provided prediction jsonl files [here][./sample_predictions] 
Sample output:

```
+---------------+-----------+-----------+----------------------+----------------------+------------------------+---------------------+
| language      |   best_em |   best_f1 |   best_answerable_em |   best_answerable_f1 |   best_unanswerable_em |   best_f1_threshold |
|---------------+-----------+-----------+----------------------+----------------------+------------------------+---------------------|
| en            |     45.39 |     51.97 |                26.65 |                36.38 |                  84.45 |               -5.95 |
| es            |     39.73 |     43.83 |                16.69 |                22.76 |                  87.75 |               -2.52 |
| zh_cn         |     32.42 |     32.43 |                 0    |                 0.01 |                  100   |              -10.53 |
| Macro Average |     39.18 |     42.74 |                14.45 |                19.72 |                  90.73 |               -6.33 |
+---------------+-----------+-----------+----------------------+----------------------+------------------------+---------------------+
```  
> Here it computes the macro average over the languages for which prediction files were supplied for.
The official macro-average requires all 26 prediction files to be included.

To run tests for the evaluation scripts
```
pytest ./tests/test_mkqa_eval.py
```


## Citation
Please cite the following if you found MKQA, our [paper](https://arxiv.org/abs/2007.15207), or these resources useful.
```
@misc{mkqa,
    title = {MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering},
    author = {Shayne Longpre and Yi Lu and Joachim Daiber},
    year = {2020},
    URL = {https://arxiv.org/pdf/2007.15207.pdf}
}
```

## License
The code in this repository is licensed according to the [LICENSE](LICENSE) file.


The Multilingual Knowledge Questions and Answers dataset is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License. To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/3.0/  


## Contact Us
To contact us feel free to email the authors in the paper or create an issue in this repository.
