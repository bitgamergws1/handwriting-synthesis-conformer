An AI model that learns to write like a human. Given a text string, it generates a realistic sequence of pen movements (dx, dy, pen-lift) that renders as natural cursive handwriting — not an image generator, but a true stroke-by-stroke sequence model.

Built with a causal Conformer backbone, Graves-style monotonic Gaussian attention (vectorized for parallel training), and a Mixture Density Network output head, trained on the IAM On-Line Handwriting Database (147 writers, 7.6M+ stroke points).
