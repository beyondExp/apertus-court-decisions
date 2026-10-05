# Apertus Court Decisions

Team beyondExp. Track 2B, Own Project / Apertus Adoption.

Code and head weights: Apache-2.0. This report: CC-BY-4.0.

## Summary

Apertus 1.5 8B stays frozen and reads the facts of a Swiss Federal Supreme Court ruling in German, French, or Italian. A sheaf head on the last hidden states returns one label from a closed list. Two tasks were trained. Legal area, thirteen official sub-areas, is the usable one. On the official test file it is correct on 88.3% of 7577 cases, against 26.0% for always choosing Substantive Criminal. At confidence 0.90 or higher it answers 72.5% of those cases and is right on 98.0% of that slice. Dismissal versus approval does not beat always saying dismissal: 80.92% against a majority baseline of 80.93%. The system is a triage aid for the legal area when it is confident. It is not a case decider and it is not legal advice.

## Architecture

The reader is swiss-ai/Apertus-v1.5-8B. Its weights are not updated. The trained module is a sheaf glue head with two maps of shape 4096 by 4096, both initialised as identity, a positive scale, and a bias on the pooled answer. Each document token and each answer token is normalised after its own map. The score of an answer is the maximum cosine between any document token and any token of that answer, plus the bias. The choice is a softmax over the closed answer list. Judgment answers are dismissal and approval. Legal-area answers are the same thirteen names on every law case.

## Target architecture

The runtime is on-premise. Scoring needs the frozen 8B weights and this head on local disk, and it does not call a network. The same layout is the air-gapped layout: after the weights are inside the boundary, inference stays there. The same image can run on a sovereign Swiss cloud host, with the weights mounted on that host rather than fetched per request. `LLM_NAME` selects the reader. `LLM_BASE_URL` and `LLM_API_KEY` are accepted by `make run` and are not used, because a chat completion does not provide the last hidden state this head scores.

## Use of Apertus

Apertus is the encoder. Training used one pass over 63328 original-language rulings, seed 1, AdamW learning rate 0.001, token cap 2048. Mean training loss was 0.527 on outcomes and 0.285 on legal area. The published checkpoint stores the head in float16. `make run` loads it, checks a forward pass on random states, and recomputes accuracy from the saved predictions.

## Data

Sources are rcds/swiss_judgment_prediction and rcds/swiss_law_area_prediction, German, French, and Italian, CC BY-SA 4.0. The input is the facts section. The label is the court's own. The reasoning section is left out. The English machine-translated split is not used. Train counts are 53474 judgments and 9854 law-area rulings. This repository does not redistribute the rulings. `data/predictions.jsonl` stores id, language, gold, prediction, confidence, and loss for the official test file and the official validation file.

## Evaluation

Official test, legal area: accuracy 88.31%, macro-F1 0.638, majority accuracy 26.04%, majority macro-F1 0.032, mean loss 0.421.

| label | support | predicted | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bankruptcy | 206 | 185 | 0.951 | 0.854 | 0.900 |
| Competition and Antitrust | 6 | 0 | 0.000 | 0.000 | 0.000 |
| Criminal Procedure | 1011 | 1319 | 0.732 | 0.955 | 0.828 |
| Employment Contract | 389 | 402 | 0.958 | 0.990 | 0.973 |
| Expropriation | 0 | 0 | 0.000 | 0.000 | 0.000 |
| Family | 263 | 160 | 1.000 | 0.608 | 0.757 |
| Intellectual Property | 11 | 7 | 1.000 | 0.636 | 0.778 |
| Other Fiscal | 1 | 0 | 0.000 | 0.000 | 0.000 |
| Public Administration | 280 | 354 | 0.305 | 0.386 | 0.341 |
| Rental and Lease | 846 | 874 | 0.960 | 0.992 | 0.976 |
| Substantive Criminal | 1973 | 1903 | 0.975 | 0.941 | 0.958 |
| Tax | 1580 | 1309 | 0.938 | 0.777 | 0.850 |
| Urban Planning and Environmental | 1011 | 1064 | 0.909 | 0.956 | 0.932 |

Official test, outcome: accuracy 80.92%, macro-F1 0.462, majority accuracy 80.93%, majority macro-F1 0.447, mean loss 0.412.

| label | support | predicted | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dismissal | 11786 | 14475 | 0.811 | 0.996 | 0.894 |
| approval | 2777 | 88 | 0.489 | 0.015 | 0.030 |

Confidence gates on the test file:

| task | threshold | n | coverage | accuracy |
| --- | ---: | ---: | ---: | ---: |
| legal area | 0.50 | 7064 | 93.2% | 92.0% |
| legal area | 0.70 | 6509 | 85.9% | 95.2% |
| legal area | 0.80 | 6203 | 81.9% | 96.6% |
| legal area | 0.90 | 5493 | 72.5% | 98.0% |
| legal area | 0.95 | 4525 | 59.7% | 99.0% |
| outcome | 0.50 | 14563 | 100.0% | 80.9% |
| outcome | 0.70 | 10684 | 73.4% | 87.8% |
| outcome | 0.80 | 6565 | 45.1% | 94.8% |
| outcome | 0.90 | 5027 | 34.5% | 97.3% |
| outcome | 0.95 | 3385 | 23.2% | 97.9% |

At confidence 0.90 or higher the outcome head answers 34.5% of test cases (5027 rulings) and every one of those predictions is dismissal, including 135 cases the court approved.

Largest legal-area confusions on the test file: Tax read as Criminal Procedure (136); Substantive Criminal read as Criminal Procedure (107); Tax read as Public Administration (99); Public Administration read as Criminal Procedure (89); Family read as Public Administration (87).

By language, test accuracy:

| task | language | n | accuracy | macro-F1 | majority accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| legal area | de | 2248 | 87.6% | 0.399 | 42.5% |
| legal area | fr | 4579 | 87.7% | 0.511 | 40.4% |
| legal area | it | 750 | 94.1% | 0.221 | 77.6% |
| outcome | de | 8270 | 80.6% | 0.466 | 80.5% |
| outcome | fr | 5569 | 81.1% | 0.457 | 81.3% |
| outcome | it | 724 | 82.7% | 0.453 | 82.7% |

Macro-F1 by language is not a ranking. The label mix differs. Italian legal-area accuracy is high because that slice is dominated by a few frequent sub-areas.

Official validation file: legal area accuracy 93.42% on 2872 cases, against majority 30.61%, macro-F1 0.685, mean loss 0.205. Outcome accuracy 79.86% on 7041 cases, against majority 79.83%, macro-F1 0.459.

## Limitations

The outcome head recalls approvals only rarely, and a high confidence score on that task is the majority label. It must not be shown as a decision. Legal area is weak on public administration, and tax cases are sometimes read as criminal procedure. Competition and antitrust, and other fiscal, have too little support to learn. Expropriation is absent from the test file. Rare classes will not appear by training the same head for another epoch. The head is not a substitute for a lawyer, a clerk, or the court.

## Reproducibility

`cd track_2b && make run` builds the image, loads `checkpoint/head.pt`, and checks that `data/predictions.jsonl` matches `data/metrics.json`. A new ruling needs a local forward of frozen Apertus 1.5 8B with hidden size 4096, then `SheafGlueHead.glue` on those states and on the embeddings of the allowed answers. Training details are in `data/metrics.json`: one epoch, seed 1, learning rate 0.001, token cap 2048.

## Next steps

Show a legal-area label only at confidence 0.90 or higher. Send the rest, and every outcome question, to a person. Review the public-administration and criminal-procedure boundary before relying on those two names.
