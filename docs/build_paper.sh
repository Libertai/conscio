#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS="$ROOT/docs"
FIGURES="$DOCS/figures"
BUILD="$DOCS/build"
FIGURE_OUT="$BUILD/figures"
FIGURE_PDF_OUT="$BUILD/figures-pdf"
PAPER="$DOCS/paper.md"
PDF_MD="$BUILD/paper.pdf.md"
PDF_OUT="$BUILD/paper.pdf"
PUBLISHED_PDF="$DOCS/paper.pdf"
CSS="$DOCS/paper.css"
GRAPHVIZ_DPI="${GRAPHVIZ_DPI:-220}"

mkdir -p "$FIGURE_OUT" "$FIGURE_PDF_OUT"

render_graphviz() {
    local input="$1"
    local output="$2"
    local format="${output##*.}"

    if ! command -v dot >/dev/null 2>&1; then
        echo "error: install Graphviz to render paper figures" >&2
        exit 1
    fi

    if [[ "$format" == "png" ]]; then
        dot -Tpng -Gdpi="$GRAPHVIZ_DPI" "$input" -o "$output"
    else
        dot -T"$format" "$input" -o "$output"
    fi
}

render_all_figures() {
    render_graphviz "$FIGURES/runtime-loop.dot" "$FIGURE_OUT/runtime-loop.svg"
    render_graphviz "$FIGURES/service-loop.dot" "$FIGURE_OUT/service-loop.svg"
    render_graphviz "$FIGURES/action-competition.dot" "$FIGURE_OUT/action-competition.svg"
    render_graphviz "$FIGURES/evaluation-ladder.dot" "$FIGURE_OUT/evaluation-ladder.svg"
    render_graphviz "$FIGURES/runtime-loop.dot" "$FIGURE_PDF_OUT/runtime-loop.png"
    render_graphviz "$FIGURES/service-loop.dot" "$FIGURE_PDF_OUT/service-loop.png"
    render_graphviz "$FIGURES/action-competition.dot" "$FIGURE_PDF_OUT/action-competition.png"
    render_graphviz "$FIGURES/evaluation-ladder.dot" "$FIGURE_PDF_OUT/evaluation-ladder.png"
}

write_pdf_markdown() {
    awk -v figure_dir="file://$FIGURE_PDF_OUT" '
        BEGIN {
            n = 0
            in_mermaid = 0
            skip_header = 1
            paths[1] = figure_dir "/runtime-loop.png"
            paths[2] = figure_dir "/service-loop.png"
            paths[3] = figure_dir "/action-competition.png"
            paths[4] = figure_dir "/evaluation-ladder.png"
            captions[1] = "Figure 1. V3 Recurrent Cognitive Runtime"
            captions[2] = "Figure 2. Persistent Service and Autonomy Loop"
            captions[3] = "Figure 3. V3 Action Competition and Execution Ordering"
            captions[4] = "Figure 4. Historical V2 Evaluation Ladder"

            print "<section class=\"cover-page\">"
            print "<div class=\"cover-kicker\">Conscio Research Draft</div>"
            print "<h1>Conscio V3</h1>"
            print "<p class=\"cover-subtitle\">An Auditable Recurrent Architecture for Machine Consciousness Research</p>"
            print "<div class=\"cover-rule\"></div>"
            print "<p class=\"cover-author\">Jonathan Schemoul<br/><span>LibertAI</span></p>"
            print "<p class=\"cover-thesis\">A text-first research instrument with private specialists, recurrent global broadcast, prospective prediction, bounded affect, independent action arbitration, restart continuity, and condition-blind intervention machinery. Historical V2 results are reported separately from the unrun V3 studies.</p>"
            print "<div class=\"cover-meta\">"
            print "<div><span>Version</span><strong>Draft v3.0</strong></div>"
            print "<div><span>Date</span><strong>July 2026</strong></div>"
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
            print "<div><span>3</span>Operational Target and Evidence Rubric</div>"
            print "<div><span>4</span>Architecture</div>"
            print "<div><span>5</span>Threat Model and Containment</div>"
            print "<div><span>6</span>Implementation Status</div>"
            print "</div>"
            print "<div class=\"toc-list\">"
            print "<div><span>7</span>Historical V2 Results and V3 Protocol</div>"
            print "<div><span>8</span>Discussion</div>"
            print "<div><span>9</span>Limitations</div>"
            print "<div><span>10</span>Future Work</div>"
            print "<div><span>11</span>Conclusion</div>"
            print "</div>"
            print "</div>"
            print "<div class=\"toc-note\">Figures and tables are generated from editable Graphviz and Markdown sources during the PDF build.</div>"
            print "</section>"
            print ""
        }
        skip_header && NR <= 4 {
            next
        }
        skip_header {
            skip_header = 0
        }
        /^\*\*Figure [1-4]\./ {
            next
        }
        /^```mermaid$/ {
            in_mermaid = 1
            n += 1
            print "<figure class=\"paper-figure\">"
            print "<img src=\"" paths[n] "\" alt=\"" captions[n] "\"/>"
            print "<figcaption>" captions[n] "</figcaption>"
            print "</figure>"
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
        --metadata pagetitle="Conscio V3: An Auditable Recurrent Architecture for Machine Consciousness Research" \
        "${pdf_engine[@]}" \
        -o "$PDF_OUT"
    cp "$PDF_OUT" "$PUBLISHED_PDF"
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
