# Paraformer-v2 PyTorch Reproduction

This workspace contains a compact PyTorch reproduction scaffold for the paper in
[`paper/samplepaper.tex`](paper/samplepaper.tex): **Paraformer-v2: An improved
non-autoregressive transformer for noise-robust speech recognition**.

The implemented core follows the paper's main architectural claim:

1. Encode acoustic features with a Conformer-style encoder.
2. Predict frame-wise CTC posteriors.
3. Compress CTC posteriors by averaging repeated non-blank alignments.
4. Project compressed posteriors into non-autoregressive decoder queries.
5. Train with `CTCLoss + CrossEntropyLoss`.

## Run The Smoke Test

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

The smoke test uses synthetic filterbank-like features and verifies that the
CTC alignment, posterior compression, decoder pass, total loss, and backward pass
all work.

## LibriSpeech Probes

Tiny local probes are available through:

```bash
PYTHONPATH=src python3 scripts/librispeech_probe.py --url dev-clean --max-utterances 16 --max-seconds 8 --epochs 1 --batch-size 2 --time-cap-seconds 900
```

For a short overfit diagnostic that emits recognizable words from real audio:

```bash
PYTHONPATH=src python3 scripts/librispeech_probe.py --url dev-clean --max-utterances 4 --max-seconds 4 --epochs 40 --batch-size 1 --alignment-mode viterbi --alignment-backend auto --eval-on-train --output runs/librispeech_probe_viterbi_overfit_metrics.json
```

That Viterbi overfit run is still only a diagnostic, but it produced word-like
predictions such as `he osn't work at all` for `he doesn't work at all` and
`hy harry builter b` for `by harry quilter m a`.

To force the training-time alignment backend during experiments, use:

```bash
--alignment-backend auto
--alignment-backend python
--alignment-backend cython
--alignment-backend triton
```

`auto` uses the Cython backend when the compiled extension is available and
prefers Triton on CUDA when available, then falls back to Cython or Python.

## Cython CTC Alignment

The CTC Viterbi path also has an optional Cython backend. The pure-Python
reference lives in
[`src/paraformer_v2/_ctc_alignment_python.py`](src/paraformer_v2/_ctc_alignment_python.py),
and the compiled backend lives in
[`src/paraformer_v2/_ctc_alignment_cython.pyx`](src/paraformer_v2/_ctc_alignment_cython.pyx).
For CUDA-capable PyTorch environments, there is also a Triton backend in
[`src/paraformer_v2/_ctc_alignment_triton.py`](src/paraformer_v2/_ctc_alignment_triton.py).

Build the extension locally with:

```bash
python3 -m venv --system-site-packages .venv-build
.venv-build/bin/pip install Cython numpy setuptools wheel
.venv-build/bin/python setup.py build_ext --inplace
```

If `nvcc` matches the CUDA version used by PyTorch, the build also produces an
optional CUDA alignment extension. If the versions do not match, `setup.py`
skips the CUDA extension and still builds the CPU Cython backend.

At runtime, `backend=auto` prefers Triton on CUDA, then the CUDA extension when
available, then CPU Cython, then pure Python.

Benchmark Python vs Cython with:

```bash
PYTHONPATH=src python3 scripts/benchmark_ctc_alignment.py --device cpu
PYTHONPATH=src python3 scripts/benchmark_ctc_alignment.py --device cuda
```

The benchmark script checks that both backends produce identical alignments
before reporting timing and speedup.

## Reproduction Target

The paper reports these central claims:

- Paraformer-v2 replaces Paraformer's CIF predictor with CTC-derived token
  embeddings.
- On LibriSpeech, Paraformer-v2 improves the 50M English NAR baseline from
  `6.5 / 10.7` WER to `3.4 / 8.0` on `test-clean / test-other`.
- On AISHELL-1, Paraformer-v2 improves the 50M NAR baseline from `5.1` test CER
  to `4.9`, with the 120M model reaching `4.7`.
- On 314 real noise samples, null-output accuracy improves from `54.5%` to
  `77.7%`.
- Inference speed remains essentially NAR-fast: reported AISHELL-1 RTF is
  `0.010` for Paraformer-v2 versus `0.011` for Paraformer and `0.254` for AR
  Conformer AED.

## Suggested Validation Plan

The cheapest useful reproduction is staged:

1. **Mechanism smoke test:** verify the CTC posterior compression and decoder
   training path on synthetic data. This is already runnable here.
2. **Small public-data probe:** train a small model on a LibriSpeech subset, such
   as `train-clean-100`, and compare CTC-only versus Paraformer-v2 decoder WER.
3. **Noise-only robustness probe:** evaluate whether CTC posterior compression
   suppresses null/noise inputs better than a CIF-style length predictor.
4. **Full benchmark:** reproduce LibriSpeech or AISHELL-1 only after compute,
   tokenizer, augmentation, and training schedule are fixed.

## Better Architecture Direction

A stronger follow-up design would keep the CTC posterior query idea, but reduce
the decoder's dependence on brittle greedy CTC boundaries:

- Use **confidence-gated posterior compression**: drop or down-weight spans whose
  non-blank posterior confidence is low, which should improve null-output
  behavior on noise.
- Add a **boundary refinement head** trained from Viterbi spans, so the model can
  split or merge CTC spans before decoder attention.
- Use **multi-resolution CTC queries** by combining a shallow CTC head and a final
  CTC head; shallow CTC tends to preserve acoustic boundary cues while final CTC
  is more semantic.
- Add a **small iterative correction decoder** for only low-confidence tokens,
  preserving most of the NAR speed while improving English subword errors.
- For streaming or low-latency settings, replace the full bidirectional decoder
  with chunk-aware attention and carry a small memory state.

These changes are narrow enough to test as ablations against the current
implementation.

### Baseline vs Better Complexity

With the same tiny LibriSpeech probe configuration (`encoder_dim=96`,
`decoder_dim=96`, `encoder_layers=3`, `decoder_layers=2`), the current baseline
and the implemented "better" variant differ as follows:

| Architecture | Parameters | Main extra components beyond shared encoder | Relative train-step cost |
| --- | ---: | --- | ---: |
| Baseline Paraformer-v2 | 1,224,153 | one CTC head, one posterior projection, one decoder pass | 1.00x |
| Better follow-up variant | 1,403,731 | shallow + final CTC heads, boundary head, richer query projection, one extra refinement decoder pass | 2.40x |

Measured train-step cost comes from a small synthetic CUDA benchmark in this
workspace using the same model dimensions as the probe config.

The better model is slower even though parameter growth is only about `14.7%`
because runtime is driven more by *where* the extra computation sits than by
parameter count alone. The encoder is unchanged, but the better variant adds:

- an extra frame-level CTC head
- a boundary prediction head over all encoder frames
- more expensive confidence-gated multi-resolution query construction
- a second decoder-style refinement pass over the compressed token queries
- extra training losses (`shallow_ctc_loss` and `boundary_loss`)

That means the model does noticeably more sequence-time work, especially in the
post-CTC path, so wall-clock time grows much more than raw parameter count.
