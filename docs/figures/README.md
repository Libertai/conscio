# Paper Figures

This directory stores editable Mermaid sources for the figures in
`docs/paper.md`.

Generate figure assets with:

```bash
docs/build_paper.sh --figures-only
```

Build the full paper PDF with:

```bash
docs/build_paper.sh
```

The build script writes generated SVGs for inspection and high-resolution PNGs
for PDF embedding under `docs/build/`, leaving these source files untouched.

PNG resolution can be tuned with environment variables:

```bash
MERMAID_PNG_SCALE=4 MERMAID_PNG_WIDTH=3200 MERMAID_PNG_HEIGHT=2000 docs/build_paper.sh
```
