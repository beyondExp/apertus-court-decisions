# Scoring

The frozen reader returns one hidden vector per document token. Each allowed answer is embedded with the same reader token embedding, then both sides pass through their own linear map. Those maps start as identity and are the trained restriction maps.

The glue score of an answer is the maximum cosine between any document token and any token of that answer. A bias term is added from the mean of the answer tokens. The logits are that score times a positive scale. The choice is the softmax over the closed set of answers.

Judgment uses the two outcome names. Legal area uses the same thirteen names on every case, so the head cannot invent a label outside that list.

A production display keeps a legal-area answer only when its softmax probability is at least 0.90. Everything else, including every outcome score, stays with a person.
