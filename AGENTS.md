# flatland benchmarl: What We're Building

We want to build a state-of-the-art torchrl flatland env for BENCHMARL to test a vareity of multi-agent alogrithms and critic models on the Flatland environment.

Your goal is to build production-grade enviroment and model architecture.

This means: no fallbacks, no hacks, no shortcuts. Production-grade, Google-quality code that at all times demonstrates a maniacal obsession with elegant minimalism.

## Public Repo Contract

- No internal-only paths, hostnames, IPs, or runbooks in tracked files.
- Keep surfaces small: no second stacks for the same use-case.

## Execution Environment

- Assume running locally on either MPS or cuda GPU. 

Our ethos: do one thing, exceedingly well — flatland env and TreeLSTM/Transformer/GNN critic models — and nothing else. Elegant minimalism isn’t just fewer lines; it’s disciplined intent plus impeccable execution.

Principles
- Small, sharp surfaces: tiny modules with crisp responsibilities; few public knobs; declarative YAML config is the source of truth.
- Explicit over magical: no hidden background machinery or side effects; contracts and control flow are obvious.
- Hot paths first: inner loops and comm paths are lean, predictable, and measured. If it doesn’t move tokens/s, stability, or correctness, it doesn’t live there.
- Fail fast, fail loud: specific guardrails with actionable remedies. No silent downshifts.
- One source of truth: one config format, one checkpoint format, one metrics schema. No duplicates to drift.
- Documentation that guides, not overwhelms: precise runbooks and remedies; zero fluff.

Craftsmanship rubric for any change
- Intent: Does this improve tokens/s, stability, or correctness?
- Uniqueness: Are we creating a second way to do something? If yes, why?
- Surface: Did we add a new public knob? Could it be expressed via existing YAML?
- Hot path: If step loop or comm changed, where is the 200‑step NVTX + ms/step delta?
- Blast radius: Did deps or coupling increase?
- Repro: Is config/provenance captured to rerun months later?
- Elegance: Is the code visibly simpler afterward?


## Research Workflow

This repo is designed to be a one-stop shop for both research and full lab-quality training. It's designed to be streamlined, opinionated, and above all elegantly minimal.


**Typical research flow:**

1. **Architecture research** — Single GPU/MPS with a small model . Needs a reasonable subset of training data to make testing meaningful.

2. **Ablations** — Compare specific changes (architecture, hyperparams, data). Can run on single GPU/MPS Should complete in hours, not days.

3. **Proxy runs** — Develop confidence around exact hyperparams for large runs.  Takes hours/1-2days.

4. **Production run** — Full training with high conviction around architecture, hyperparams. Takes hours.



## Contract (Principles)

- Config: YAML-only. CLI reads YAML with environment overrides; 
- Data: no hardcoded locations; all paths provided via YAML. Support robust dataset mixing (per‑dataset weights/temperatures), deterministic sharding, and exact resume (RNG + shard/offset).
- Metrics/Tracking: Tensorboard and wandb and log to stdout.
Purpose
- No fallbacks, hacks, or shortcuts; correctness and performance are first‑class.

References
- Current standard: BenchMARL (torchrl ENV, model and algorithms).
- Inspiration: PufferRL (speed, maximize SPS, SOTA on algos/sweeping (CARBS, Muon optimizer)

