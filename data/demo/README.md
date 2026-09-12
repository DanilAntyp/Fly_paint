# Demo fixtures

`cat.png`, `leaf.png` and `blank.png` are procedurally authored project fixtures,
reproducible using `flypaint.targets.fixture` (blank is white RGB 64×64).
They contain no third-party image content and no neural data.

Synthetic mode generates a seeded sparse random graph in `load_graph()` and
always labels it **Synthetic demo graph**. It is never substituted after a
real-graph load failure. Real extracted graphs live in ignored `data/processed`.
