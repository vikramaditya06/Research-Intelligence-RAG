# Evaluation dataset

No benchmark questions are included in this repository because a valid
retrieval benchmark requires **real documents and real chunk labels** from the
same indexed corpus. Fabricating UUIDs or placeholder questions would make the
reported retrieval metrics misleading.

## Create a real benchmark

1. Choose a fixed set of research PDFs.
2. Ingest those PDFs through the normal `/documents/upload` flow.
3. Select representative research questions whose answers are present in the
   indexed corpus.
4. Manually label the relevant child chunk IDs and, where useful, parent IDs.
5. Replace the empty `sample_dataset.json` with those real labelled examples.
6. Run `python scripts/evaluate.py` against that fixed corpus.

Each example must use IDs returned by the **same database/corpus** used for
evaluation:

```json
[
  {
    "question": "What limitation does the proposed method have?",
    "document_ids": ["<real-document-uuid>"],
    "relevant_chunk_ids": ["<real-chunk-uuid>"],
    "relevant_parent_ids": ["<real-parent-uuid>"]
  }
]
```

`document_ids`, `relevant_chunk_ids`, and `relevant_parent_ids` must not be
invented or copied from another database.

The evaluator skips examples without labels and does not claim benchmark
results when there are no real labelled examples.
