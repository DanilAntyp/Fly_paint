"""Summarize saved measurements; no new training or selectively chosen seeds."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    exp=json.loads((ROOT/'docs/EXPERIMENTS.json').read_text())
    ablation=json.loads((ROOT/'docs/ABLATION.json').read_text())
    bench=json.loads((ROOT/'docs/BENCHMARKS.json').read_text())
    lines=['# Observed prototype results — 2026-09-12','',
           'Linux x86_64, Python 3.12.14; **not measured on an M1**. Source data was',
           'downloaded successfully. Training, API and browser all executed. Image',
           'copying remains weak: these are attempts, not reliable reconstructions.','',
           '## Per-image learning','',
           'Procedural cat image, 128×128 target, two training starts, three held-out',
           'starts per checkpoint; 16 candidates × 10 generations × 256 steps.',
           'Three independent initializations (42/43/44) for each graph. Evaluation',
           'seed windows overlap across checkpoints, so rows are not nine independent',
           'images or fully independent evaluation starts. All checkpoints are shown.','',
           '| Graph | Training seed | Initial train score | Final train score | Untrained held-out | Trained held-out | Random actions | Trained coverage | Trained F1 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in exp['records']:
        mean=lambda label,key:float(np.mean([x[key] for x in r['held_out'][label]]))
        lines.append(f"| {r['graph']} | {r['seed']} | {r['training_initial']:.2f} | {r['training_final']:.2f} | {mean('untrained','score'):.2f} | {mean('trained','score'):.2f} | {mean('random_actions','score'):.2f} | {100*mean('trained','recall'):.1f}% | {mean('trained','f1'):.3f} |")
    lines+=['', 'Real seed 42 improves, but real seed 44 underperforms random actions on the',
            'held-out starts. Real seed 43 improves score while losing a little coverage.',
            'The objective can favor fewer penalties over coverage. There is no strong',
            'evidence of a real-connectome advantage over the matched random graph.',
            'The random control matches nodes, edge count and sensory/output adapters;',
            'it does not preserve degrees, connection-weight distributions or cell types.','',
            'These are per-image adaptations; no arbitrary-image zero-shot success is',
            'claimed. Leaf and circle are held-out target fixtures in the frozen-model',
            'ablation, not additional training targets. A portrait dataset was not tested.','',
            '![Actual target and frozen checkpoint attempts](images/attempts.png)','',
            'The comparison uses held-out start 1042 of checkpoint 42. All visible ink',
            'comes from bounded controller movements. The untrained and trained images',
            'are incomplete; no tracing teacher or image-generation output is used.','',
            '## Group ablation','',
            '27 paired episodes per mask: 3 independently trained checkpoints × 3 targets',
            '(adapted cat, held-out leaf, held-out circle) × 3 held-out starts. The four',
            'groups contain 67 visual_projection, 270 CX, 158 descending_neuron and',
            '17 vnc_motor nodes. All six pairs were also tested. Weights, normalization',
            'and readouts stayed frozen. See [raw rows](ABLATION.json).','',
            '| Disabled group | Mean F1 | Mean coverage | Mean off-target pixels | Mean task score | Mean step ms | All paired losses ≤5%? |',
            '|---|---:|---:|---:|---:|---:|---|']
    groups=defaultdict(list)
    for run in ablation['runs']:
        for row in run['rows']:
            if row['mask'] in ['intact',*ablation['pruning_decisions']]:groups[row['mask']].append(row)
    for label,rows in groups.items():
        mean=lambda key:float(np.mean([r[key] for r in rows]))
        passed='control' if label=='intact' else str(ablation['pruning_decisions'][label]['all_paired_runs_within_5_percent'])
        lines.append(f"| {label} | {mean('f1'):.4f} | {100*mean('recall'):.2f}% | {mean('off_target'):.1f} | {mean('score'):.2f} | {mean('step_ms'):.3f} | {passed} |")
    lines+=['', 'Some mean metrics improve after masking CX or motor nodes, but individual',
            'paired cases fail the predeclared threshold. **No additional pruning is',
            'justified or deployed.** A mask retains the same sparse allocation and is',
            'not a memory optimization. Peak RSS for the complete ablation process was',
            f"{ablation['peak_rss_mib']:.1f} MiB; per-group memory savings were not measured.",'',
            '## Cost and graph sizes','',
            '| Nodes | Edges | Full CPU step | Process peak RSS |',
            '|---:|---:|---:|---:|']
    for r in bench:lines.append(f"| {r['nodes']} | {r['edges']:,} | {r['step_ms']:.3f} ms | {r['peak_rss_mib']:.1f} MiB |")
    lines+=['', 'The timings include observations, reservoir, motion and rasterization, not',
            'optimization or browser rendering. Larger graphs have not been trained for',
            'quality comparison. Source extraction used around 1.4 GiB peak RSS. Synthetic',
            '512-node reservoir-only timing was 58,149 steps/s at 42.1 MiB; this narrower',
            'benchmark is not training throughput. Some timings occurred alongside other',
            'validation work and are observational, not isolated hardware certification.','',
            '## Verification actually run','',
            '- Fresh Python 3.12 virtual environment installation from requirements-lock.txt.',
            '- 16 unittest tests: graph orientation/IDs/weights, image formats/limits/EXIF/alpha,',
            '  aspect ratio, blank targets, reward exploits, bounds, reproducible learning,',
            '  masks, directed extraction, endpoint adapters, pause/cancel, replacement,',
            '  WebSocket completion and PNG export.',
            '- compileall on src, scripts, services and tests.',
            '- TypeScript noEmit check and production Vite build.',
            '- Two Chromium Playwright scenarios on the real task graph: full image flow,',
            '  progressive ink, pause, completed training, PNG download, reset/replacement,',
            '  invalid/blank images, detail changes and a 390-pixel mobile viewport.',
            '- Successful download of all locked ARM64/universal wheels for Python 3.12',
            '  on macOS 14+. Native M1 execution, battery use and thermals remain untested.','',
            'CI runs synthetic mode without downloading the 1 GB source dataset. Real-mode',
            'evidence here comes from local verification, not an asserted CI result.','',
            '## Reproduce experiments','', '```bash',
            'OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_experiments.py \\',
            '  --graph data/processed/task-512',
            'OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_ablation.py \\',
            '  --graph data/processed/task-512 \\',
            '  --checkpoints runs/real-cat-42.npz runs/real-cat-43.npz runs/real-cat-44.npz',
            'OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/extract_task_graph.py \\',
            '  --nodes 1024 --output data/processed/task-1024',
            'OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/extract_task_graph.py \\',
            '  --nodes 2048 --output data/processed/task-2048',
            'OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/measure.py --graph data/processed/task-512',
            '```','',
            'Next quality work should improve exploration and test broader image/starting',
            'pose curricula. More generations alone can overfit. Revisit pruning only',
            'after the intact controller reaches useful drawing quality.']
    (ROOT/'docs/RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
