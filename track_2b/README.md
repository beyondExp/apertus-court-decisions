# Apertus Court Decisions

Track: Own Project / Apertus Adoption (track 2B). Team: beyondExp.

Swiss Federal Supreme Court rulings already carry an official label. Apertus 1.5 8B reads the facts in German, French, or Italian. A sheaf head on the last hidden states returns one closed label. The useful label is the legal sub-area, one of thirteen official names. Dismissal or approval was trained and evaluated, and it does not beat always saying dismissal.

Apertus stays frozen. Only the head is trained, for one pass over 63,328 original-language rulings. The reasoning section is not part of the input. The label is the court's own.

## Run

From this directory:

```bash
make run
```

Docker builds an image, loads `checkpoint/head.pt`, checks that the head forward runs, and recomputes accuracy from `data/predictions.jsonl`. The check fails if those predictions disagree with `data/metrics.json`.

Environment variables accepted by the template:

- `LLM_NAME` defaults to `swiss-ai/Apertus-v1.5-8B`
- `LLM_BASE_URL` and `LLM_API_KEY` are accepted and unused

The head cannot be driven from chat completions. It needs the last hidden state of the frozen 8B reader, plus the token embeddings of the allowed answers. Once those weights and this head are on disk, scoring does not use the network. That is the on-premise and air-gapped path.

## Result

On the official test file, legal area is correct for 88.3% of 7,577 cases. Always choosing the most common sub-area is correct for 26.0%. At a confidence of at least 0.90 the head answers 72.5% of cases and is right on 98.0% of that slice.

Outcome accuracy on the same file is 80.9%, the same as always saying dismissal. That score is not a decision.

The full tables are in `technical_report.md` and `beyondExp_Report.pdf`.

## Data

`data/` holds the metrics and the per-case predictions. It does not hold the rulings. Rebuild the training file from:

- `rcds/swiss_judgment_prediction` (de, fr, it)
- `rcds/swiss_law_area_prediction` (de, fr, it)

Both are CC BY-SA 4.0. Facts only. Judgment options are `dismissal` and `approval`. Law options are the thirteen sub-area names on every law row. The English `mt_en` split is not used.

## Licenses

Code and `checkpoint/head.pt` are Apache-2.0. `technical_report.md` and `beyondExp_Report.pdf` are CC-BY-4.0.
