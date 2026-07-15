# Paper Figures

This directory stores editable Mermaid diagrams used in `docs/paper.md` and
matching Graphviz sources used for deterministic PDF rendering.

Generate figure assets with:

```bash
docs/build_paper.sh --figures-only
```

Build the full paper PDF with:

```bash
docs/build_paper.sh
```

The build script writes generated SVGs for inspection and high-resolution PNGs
for PDF embedding under `docs/build/`, then publishes the finished artifact as
`docs/paper.pdf`. Mermaid sources remain editable and untouched.

The V3 PDF figure set is:

- `runtime-loop.dot` - recurrent specialist and broadcast cycle;
- `service-loop.dot` - persistence, policy, and restart recovery;
- `action-competition.dot` - proposal competition and execution ordering;
- `evaluation-ladder.dot` - historical V2 baseline ladder.

PNG resolution can be tuned with an environment variable:

```bash
GRAPHVIZ_DPI=300 docs/build_paper.sh
```
