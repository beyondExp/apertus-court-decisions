# Apertus Court Decisions

A frozen Apertus 1.5 8B reader and a small sheaf head for Swiss Federal Supreme Court rulings. The head returns a legal sub-area when it is confident. It does not decide dismissal or approval.

This repository is the track 2B submission. The project lives in `track_2b/`.

```bash
cd track_2b
make run
```

`make run` builds the Docker image, loads the published head, and recomputes the official scores from `data/predictions.jsonl`. A live score on a new ruling needs the local Apertus 8B weights, because the head reads last-layer hidden states.

Code and head weights are Apache-2.0. The technical report is CC-BY-4.0. The source rulings are not in this repository. They remain on `rcds/swiss_judgment_prediction` and `rcds/swiss_law_area_prediction` under CC BY-SA 4.0.
