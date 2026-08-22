# Handwriting Synthesis Conformer

An AI model that learns to write like a human. Given a text string, it generates a realistic sequence of pen movements (`dx`, `dy`, pen-lift) that renders as natural cursive handwriting — not an image generator, but a true stroke-by-stroke sequence model.

Built with a causal Conformer backbone, Graves-style monotonic Gaussian attention (vectorized for parallel training), and a Mixture Density Network output head, trained on the IAM On-Line Handwriting Database (147 writers, 7.6M+ stroke points).

---

## How It Works

```
strokes_in (pen history)
      |
      v
StrokeEmbedding
      |
      v
Causal Conformer Backbone   <---   text  -->  Text Encoder
      |          (the "hand")                       |
      |                                              v
      |                              Monotonic Gaussian Attention (the "eye")
      |                                              |
      |                                       context vector
      v                                              |
      +--------------------- concat ------------------+
                          |
                          v
                    Fusion Layer
                          |
                          v
                  MDN Head (predicts a probability
                  cloud of possible next movements)
```

- **Causal Conformer backbone** — combines self-attention and convolution to capture both the long-range flow and the local, letter-level curvature of pen movement, without ever looking ahead at future timesteps.
- **Monotonic Gaussian attention** (Graves, 2013) — reads the input text strictly left-to-right, never skipping or repeating a character. Re-implemented as a fully vectorized `cumsum`-based operation so the whole sequence trains in parallel instead of one timestep at a time.
- **Mixture Density Network head** — instead of one fixed answer, predicts a mixture of Gaussians at every timestep, capturing the natural variation in how a human draws the same letter twice.

## Repository Structure

```
.
├── images/                                        # generation-check outputs, diagnostic charts
├── handwriting_ai_documentation-Hinglish_Version.md   # full build log, in Hinglish
├── handwriting_ai_documentation_english.md            # full build log, in English
└── README.md
```

## Full Build Log

This project was built (and debugged) iteratively across several training rounds, with every bug, hypothesis, and fix documented as it happened — not cleaned up after the fact. If you want the complete story of how this went from broken output to a working model, including every dead end:

- 📄 [Full documentation — English](./handwriting_ai_documentation_english.md)
- 📄 [Full documentation — Hinglish](./handwriting_ai_documentation-Hinglish_Version.md)

Highlights from the journey:

| Round | What Happened |
|---|---|
| 1–2 | Base training + exposure bias and pen-lift fixes |
| 3 | Attention "blurry vision" bug found and fixed (beta floor) |
| 4 | SGDR warm restarts + entropy regularizer; attention confirmed healthy via heatmap measurement |
| 5 | Root cause of character-level errors traced to training-data frequency imbalance; rarity-weighted loss added |
| 6 (in progress) | Tuning rarity-weight ceiling and restart schedule to fix attention instability introduced in Round 5 |

## Current Status

The model reliably learns handwriting *physics* — smooth cursive strokes, correctly timed pen-lifts, and stable monotonic attention across the full text. The open problem is **character identity for rare letters** (especially capitals): letters that appear infrequently in the IAM-OnDB training corpus (a proper-noun capital like "N", for example) are still misrendered even after targeted rarity-weighting, and current work is on stabilizing that fix without destabilizing attention.

## Dataset

[IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) — 147 writers, ~12,100 usable sequences, 7.6M+ raw stroke points after preprocessing.

## Training

```bash
python train_expert.py \
  --npz_path /path/to/iam_ondb_processed.npz \
  --warm_start_from /path/to/checkpoint.pth \
  --checkpoint_dir ./checkpoints \
  --epochs 40 --batch_size 32 \
  --lr_schedule warm_restarts --restart_cycle_epochs 8 \
  --use_char_rarity_weighting \
  --gen_check_interval 5
```

See the full documentation for the complete set of tunable flags (rarity-weighting range, restart decay, generation-check settings, etc.) and what each one is for.
