# Theory Mapping and References

Conscio's public face is the consciousness layer for LLM agents. This page
keeps the research mapping and source trail close by without making the README
open like a paper.

## Theory Mapping

| Theory | Conscio implementation | Evidence status |
| --- | --- | --- |
| Global Workspace Theory / GNW | Attention competition; broadcast winners assemble the model context | Gating ablation refuted on task scores so far |
| Higher-Order / Self-Model theories | Live self-state computed from measured signals, fed back into the prompt | Largest ablation effect on one model; grounds self-report |
| Attention Schema Theory | Runtime model of focus, ignored candidates, dispersion | Recorded per tick |
| Predictive Processing | Expectations registered before execution, resolved against results | Confirmed on one model, inconclusive on the other |
| Memory theories of agency | Provenance-tiered facts, hybrid retrieval, budgeted consolidation | Confirmed ablation on both models |
| Autopoietic/agentic framing | Drives with satiation, durable goals, self-review, autonomous VM action | Long-horizon suite, mixed by model |

## Research Claim

Conscio makes an operational claim, not a phenomenal one: a computational
organization with persistent self-modeling, budgeted global attention,
provenance-tracked memory, appraisal, goal formation, reflection, and
autonomous action, where each mechanism is a feature flag whose contribution
is measured by ablation, and where the agent's self-description is scored
against its own traces.

The paper in [docs/paper.md](../paper.md) states the criteria, threat model,
and results, including the negative ones.

## References

- Butlin et al. 2023, "Consciousness in Artificial Intelligence":
  https://arxiv.org/abs/2308.08708
- Albantakis et al. 2023, "Integrated Information Theory 4.0":
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465
- Webb & Graziano 2015, "The Attention Schema Theory":
  https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00500/full
- Friston 2010, "The free-energy principle":
  https://www.nature.com/articles/nrn2787
- LIDA Global Workspace architecture:
  https://www.aaai.org/Library/Symposia/Fall/2007/fs07-01-011.php
