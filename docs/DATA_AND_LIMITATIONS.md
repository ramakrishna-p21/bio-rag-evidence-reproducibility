# Data availability and limitations

## What this repository contains

Software, configuration, tests on synthetic fixtures, the R analysis programs with their locked
environment, plotting code, and the researcher approval records that bind protected runs to exact
code revisions. It does not contain study data or results.

## Data availability

| Material | In this repository | Notes |
| --- | --- | --- |
| Source publications (JATS XML, PDF) | No | Publication licenses differ by article and are not assumed to allow redistribution |
| Corpus manifest and release records | No | |
| Benchmark questions, gold answers and evidence annotations | No | Withheld to keep the held-out test questions usable |
| Generated answers, contexts, attempts, receipts | No | Preserved privately |
| Review and adjudication records | No | Preserved privately |
| Statistical result tables and figures | No | Not yet approved for release |
| Model weights | No | Public models, pinned by revision in `configs/` |
| Code, configs, tests, locks, approvals | Yes | |

A few short text fragments from the article PMC13488024 (licensed CC-BY-4.0 in its JATS permissions) appear in unit-test fixtures; they are attributed here and are not final-test benchmark material.

No decision has yet been made on publishing any of the withheld material, on the license for the
software, or on an archival deposit. When such decisions are taken they will be recorded here. Until
then, the workflows that need controlled data cannot be repeated by outside readers, and the
repository should not be read as a complete public replication package.

## Evaluation scope

The final test generated 4,575 observations. Only the 1,830 confirmatory observations
(305 questions × 2 representations × 3 repeats) were evaluated, under an approved amendment to the
evaluation protocol. The closed-book baseline (915 observations) and the dense-retrieval,
192-token-chunk and full-corpus arms (1,830 observations) were generated and preserved but not
evaluated. They are deferred work. Nothing in the study's results describes them.

## Custody exception

The final-test batch was produced in a cloud notebook session. The original external custody
requirement for the run (called G13 in the project's records) could not be established: the
original external anchor and execution root were held on the notebook runtime and were lost with it.
The requirement remains **not established**.

The retained batch was instead checked after the fact. The downloaded copy matched its manifest file
by file, its internal write-once seal chain verified, and it matched the inventory of the request
that named it. That shows the copy is complete and internally consistent. It does not show that the
batch was independently retained or that it was not replaced before the request was written. The
researcher approved an exception that allows the batch to be evaluated as the batch of record on this
basis, and the exception and its residual risk are part of the scientific record and are reported
with the results. `approvals/final_test_evaluation.json` binds the evaluation to that exception by
hash.

## Approval records authorize historical revisions

`approvals/` holds records that name the exact source revision a protected step was approved for.
They authorize those historical revisions, not the consolidated revision of this repository. This
revision has no approval of its own, and protected commands that require an approved clean revision
refuse it. The revisions are listed in [REPRODUCING.md](REPRODUCING.md#6-historical-revisions-and-checkouts).
Approval files that were superseded during the work are kept privately and are not current authority.

## Scientific limitations

1. **Single reviewer.** Stage A, B and C review judgments come from one researcher. There is no
   second rating and no reliability statistic.
2. **AI assistance.** Where AI-generated proposals were used to prepare review material, they were
   advisory, built from masked material, and the researcher decided every label. Part of the early
   Stage B review had an isolation deviation, disclosed in the evaluation scope amendment.
3. **Masked, not blinded.** Review packets hide representation names and are order-randomized, but
   extraction artifacts remain visible and can reveal the source form. A representation guess was
   recorded, and an aggregate Stage A summary was exposed to the reviewer after judgments were made.
4. **Scope.** Only the confirmatory arm was evaluated (above).
5. **Generator and corpus.** One open-weight generator (Qwen3-8B), one pinned prompt, 50 papers, two
   representations, one PDF converter (Docling) and one search method in the confirmatory arm.
   Conclusions are conditional on these choices and are not claims about other models, parsers or
   corpora.
6. **Repeats are replay checks.** Decoding was greedy, so the three repeats test computational
   replay. They do not estimate stochastic generation variability.
7. **Output length.** Generations were limited to 128 new tokens, and some reached that limit.
8. **Environment.** The generation environment lock differs from the one used in qualification in
   project packages and the BM25 package. Dense-retrieval ranks were computed on CPU in the cloud
   session, and identity across hardware was not established. The final R analysis result is
   compared byte for byte and has been reproduced only on one platform.
9. **Development pilots.** Earlier development pilots used earlier chunking, prompt and scoring
   versions and are exploratory, not findings.
10. **Table coverage.** Table rows are only partly indexed at the chosen chunk size in development
    checks, and whether evidence in tables and in prose is covered equally has not been established.
11. **Cloud use.** The use of a personal cloud notebook and storage account for the final run rests on
    the researcher's own decision and is disclosed as such.
