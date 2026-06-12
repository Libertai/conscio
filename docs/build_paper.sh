#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS="$ROOT/docs"
FIGURES="$DOCS/figures"
BUILD="$DOCS/build"
MERMAID_OUT="$BUILD/figures"
MERMAID_PDF_OUT="$BUILD/figures-pdf"
PUPPETEER_CONFIG="$FIGURES/puppeteer-config.json"
PAPER="$DOCS/paper.md"
PDF_MD="$BUILD/paper.pdf.md"
PDF_OUT="$BUILD/paper.pdf"
CSS="$DOCS/paper.css"
MERMAID_PNG_SCALE="${MERMAID_PNG_SCALE:-3}"
MERMAID_PNG_WIDTH="${MERMAID_PNG_WIDTH:-2400}"
MERMAID_PNG_HEIGHT="${MERMAID_PNG_HEIGHT:-1600}"

mkdir -p "$MERMAID_OUT" "$MERMAID_PDF_OUT"

render_mermaid() {
    local input="$1"
    local output="$2"
    shift 2
    local extra_args=("$@")

    if command -v mmdc >/dev/null 2>&1; then
        mmdc -i "$input" -o "$output" -b transparent -t neutral -p "$PUPPETEER_CONFIG" "${extra_args[@]}"
        return
    fi

    if command -v npx >/dev/null 2>&1; then
        npx -y @mermaid-js/mermaid-cli -i "$input" -o "$output" -b transparent -t neutral -p "$PUPPETEER_CONFIG" "${extra_args[@]}"
        return
    fi

    echo "error: install mermaid-cli first: npm install -g @mermaid-js/mermaid-cli" >&2
    exit 1
}

render_all_figures() {
    render_mermaid "$FIGURES/runtime-loop.mmd" "$MERMAID_OUT/runtime-loop.svg"
    render_mermaid "$FIGURES/service-loop.mmd" "$MERMAID_OUT/service-loop.svg"
    render_mermaid "$FIGURES/attention-weights.mmd" "$MERMAID_OUT/attention-weights.svg"
    render_mermaid "$FIGURES/evaluation-ladder.mmd" "$MERMAID_OUT/evaluation-ladder.svg"
    local png_args=(-s "$MERMAID_PNG_SCALE" -w "$MERMAID_PNG_WIDTH" -H "$MERMAID_PNG_HEIGHT")
    render_mermaid "$FIGURES/runtime-loop.mmd" "$MERMAID_PDF_OUT/runtime-loop.png" "${png_args[@]}"
    render_mermaid "$FIGURES/service-loop.mmd" "$MERMAID_PDF_OUT/service-loop.png" "${png_args[@]}"
    render_mermaid "$FIGURES/attention-weights.mmd" "$MERMAID_PDF_OUT/attention-weights.png" "${png_args[@]}"
    render_mermaid "$FIGURES/evaluation-ladder.mmd" "$MERMAID_PDF_OUT/evaluation-ladder.png" "${png_args[@]}"
}

write_pdf_markdown() {
    awk '
        BEGIN {
            n = 0
            in_mermaid = 0
            skip_header = 1
            images[1] = "![Figure 1. Per-Episode Cognitive Runtime](docs/build/figures-pdf/runtime-loop.png)"
            images[2] = "![Figure 2. Persistent Service and Autonomy Loop](docs/build/figures-pdf/service-loop.png)"
            images[3] = "![Figure 3. Base Attention Score Weights](docs/build/figures-pdf/attention-weights.png)"
            images[4] = "![Figure 4. Evaluation Ladder](docs/build/figures-pdf/evaluation-ladder.png)"

            print "<section class=\"cover-page\">"
            print "<div class=\"cover-kicker\">Conscio Research Draft</div>"
            print "<h1>Conscio</h1>"
            print "<p class=\"cover-subtitle\">An Operational Architecture and Evaluation for Auditable Machine Consciousness</p>"
            print "<div class=\"cover-rule\"></div>"
            print "<p class=\"cover-author\">Jonathan Schemoul<br/><span>LibertAI</span></p>"
            print "<p class=\"cover-thesis\">A persistent agent runtime for studying operational consciousness through inspectable mechanisms: attention, self-modeling, memory, appraisal, goals, prediction error, reflection, and autonomous action — with baseline-ladder, ablation, and trace-grounded self-report results on two model families.</p>"
            print "<div class=\"cover-meta\">"
            print "<div><span>Version</span><strong>Draft v2.0</strong></div>"
            print "<div><span>Date</span><strong>June 2026</strong></div>"
            print "<div><span>Affiliation</span><strong>LibertAI</strong></div>"
            print "</div>"
            print "</section>"
            print ""
            print "<section class=\"toc-page\">"
            print "<h1>Contents</h1>"
            print "<div class=\"toc-grid\">"
            print "<div class=\"toc-list\">"
            print "<div><span>1</span>Introduction</div>"
            print "<div><span>2</span>Background and Related Work</div>"
            print "<div><span>3</span>Operational Definition</div>"
            print "<div><span>4</span>Architecture</div>"
            print "<div><span>5</span>Threat Model and Containment</div>"
            print "<div><span>6</span>Implementation Status</div>"
            print "</div>"
            print "<div class=\"toc-list\">"
            print "<div><span>7</span>Evaluation and Results</div>"
            print "<div><span>8</span>Discussion</div>"
            print "<div><span>9</span>Limitations</div>"
            print "<div><span>10</span>Future Work</div>"
            print "<div><span>11</span>Conclusion</div>"
            print "</div>"
            print "</div>"
            print "<div class=\"toc-note\">Figures and tables are generated from editable Mermaid and Markdown sources during the PDF build.</div>"
            print "</section>"
            print ""
        }
        skip_header && NR <= 4 {
            next
        }
        skip_header {
            skip_header = 0
        }
        /^```mermaid$/ {
            in_mermaid = 1
            n += 1
            print images[n]
            next
        }
        in_mermaid && /^```$/ {
            in_mermaid = 0
            next
        }
        !in_mermaid {
            print
        }
    ' "$PAPER" > "$PDF_MD"
}

build_pdf() {
    if ! command -v pandoc >/dev/null 2>&1; then
        echo "error: install pandoc to build docs/build/paper.pdf" >&2
        exit 1
    fi

    write_pdf_markdown
    local pdf_engine=()
    if command -v weasyprint >/dev/null 2>&1; then
        pdf_engine=(--pdf-engine=weasyprint --css "$CSS")
    elif command -v xelatex >/dev/null 2>&1; then
        pdf_engine=(--pdf-engine=xelatex)
    elif command -v pdflatex >/dev/null 2>&1; then
        pdf_engine=(--pdf-engine=pdflatex)
    else
        echo "error: install weasyprint, xelatex, or pdflatex to build docs/build/paper.pdf" >&2
        exit 1
    fi

    pandoc "$PDF_MD" \
        --from gfm+raw_html \
        --standalone \
        --resource-path "$DOCS" \
        --metadata pagetitle="Conscio: An Operational Architecture and Evaluation for Auditable Machine Consciousness" \
        "${pdf_engine[@]}" \
        -o "$PDF_OUT"
}

case "${1:-}" in
    --figures-only)
        render_all_figures
        ;;
    --help|-h)
        echo "usage: docs/build_paper.sh [--figures-only]"
        ;;
    "")
        render_all_figures
        build_pdf
        ;;
    *)
        echo "usage: docs/build_paper.sh [--figures-only]" >&2
        exit 2
        ;;
esac
