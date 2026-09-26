# AdaEva: Accelerating LLM-Driven Algorithm Design with Adaptive Partial Evaluation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Review](https://img.shields.io/badge/Review-Double--Blind-lightgrey.svg)]()

> Anonymous code release accompanying the ICLR submission *"AdaEva: Accelerating LLM-Driven Algorithm Design with Adaptive Partial Evaluation"*. All author and institution information has been removed for double-blind review.

---

## Overview

LLM-driven automated algorithm design (**LLM4AD**) frameworks such as **EoH**, **HiFo-Prompt** and **LLaMEA** evaluate every generated heuristic $h$ on the entire training pool $\mathcal{I}=(x_1,\dots,x_K)$ (uniform allocation, $b_h^{(t)} = K$). In expensive domains such as routing, packing, scheduling and black-box optimisation, **candidate evaluation, not LLM inference, is the computational bottleneck**.

**AdaEva** is an instance-aware, adaptive evaluation layer that is dropped in between candidate generation and environmental selection:

- **Preserved:** the LLM prompting engine, parent selection, code mutation/crossover operators and the per-instance simulator $f(h;x)$ of each host framework remain unchanged.
- **Intercepted:** only the evaluation of the candidate pool $\mathcal{U}^{(t)} = \Pi^{(t)} \cup \mathcal{O}^{(t)}$. Instead of a full grid, AdaEva adaptively allocates per-candidate budgets $b_h^{(t,r)}$ over progressive steps $r$ and discards unpromising candidates early.

Two instantiations are provided:

| Variant | Mechanism | Code |
| :--- | :--- | :--- |
| **AdaEva-R** | Statistical iterated racing on a blocked instance prefix (Friedman omnibus + Conover or Nemenyi post-hoc test, or paired *t*-test) | [`src/racing/`](src/racing/) |
| **AdaEva-S** | Cache-aware successive halving over geometrically growing instance budgets | [`src/sh/`](src/sh/) |

These are compared against two static baselines, also included:

| Baseline | Allocation | Code |
| :--- | :--- | :--- |
| **Uniform-$k$** (static partial evaluation) | every candidate on a fixed subset of $k$ instances ($k = 1$ gives Uniform-1) | [`src/fixed/`](src/fixed/) |
| **Uniform-$K$** (full evaluation) | every candidate on the whole training pool | [`src/full/`](src/full/) |

**Host frameworks × domains.** EoH and HiFo-Prompt are run on three combinatorial problems: Online Bin Packing (**OBP**), TSP with guided local search (**TSP-GLS**) and Flow-Shop Scheduling with GLS (**FSSP-GLS**). LLaMEA is run on continuous black-box optimisation (**BBOB**, 24 noiseless functions in 5-D).

---

## Notation

The code follows the notation of the paper:

| Symbol | Meaning | Where it appears in the code |
| :--- | :--- | :--- |
| $\mathcal{I} = (x_1,\dots,x_K)$ | Shared, ordered training instance pool | `--n-instances` ($K$); instance order printed at start-up by `print_instance_order` |
| $f(h;x)$ | Cost of heuristic $h$ on instance $x$ (minimised) | host evaluator from `packages/LLM4AD`, `packages/HiFo-Prompt` or `packages/LLaMEA` |
| $\hat g(h;n)$ | Empirical mean cost on the prefix $x_1,\dots,x_n$ | `Config.costs_by_inst` in [`src/utils/config.py`](src/utils/config.py) |
| $B$ | Global budget of candidate–instance evaluations | `--budget-cap`; defaults to `pop_size × ref_max_generations × n_instances` (the cost of Uniform-$K$ over the reference horizon) |
| $M$ | Population size | `--pop-size` (EoH/HiFo), `--n-parents` (LLaMEA) |
| $\mathcal{U}^{(t)}$ | Pool at generation $t$: parents ∪ offspring | `combined` in [`src/racing/base.py`](src/racing/base.py) |
| $\mathcal{U}^{(t,r)}$ | Contenders alive at racing step $r$ | `Config.alive` |
| $\mathcal{V}^{(t)}$ | Survivors of the final racing step | `survivors` returned by `elitist_race` |
| $\hat h^{(t)}$ | Logged incumbent of generation $t$ | `select_incumbent` in [`src/racing/base.py`](src/racing/base.py) |

---

## Method

### AdaEva-R: statistical racing

Implemented in [`src/utils/race.py`](src/utils/race.py) (`elitist_race`) and [`src/utils/tests.py`](src/utils/tests.py). It follows the elitist race of irace.

1. **Blocked prefix.** All contenders are evaluated on the same instance sequence $x_1, x_2, \dots$ (with a per-instance seed shared across candidates), so comparisons are paired and across-instance variance is blocked out.
2. **Carry-over and reuse.** Costs are cached per `(candidate, instance)` pair. Parents carried over from earlier generations reuse their stored costs; only *fresh* evaluations are charged against $B$. `--elitist-new-instances` new instances are added to each race.
3. **Elimination test.** After `--t-first` instances, and then every `--t-each` instances:
   - `--test-type friedman` (default): build the rank matrix on the shared prefix (average ranks for ties) and run the Friedman omnibus test at level `--alpha`. If it rejects, run a post-hoc comparison against the best-ranked contender: `--posthoc-test-type conover` (default; Conover as in irace) or `nemenyi` (critical-difference test).
   - `--test-type ttest`: paired *t*-tests against the best contender.
   - Contenders that are significantly worse are eliminated.
4. **Stopping.** A race stops when the number of survivors reaches $M$, when the instance pool is exhausted, when the budget is exhausted, or after `--elitist-limit` consecutive tests that eliminate nothing.
5. **Refill (optional, `--save-pop`).** If fewer than $M$ candidates survive, the population is topped up with the best eliminated candidates. Refilled candidates keep `alive = False`, so they are never reported as the incumbent.

Crashed candidates are rejected by default (`--deal-with-crashed rejection`) or given a large penalty cost (`penalty`). Candidates that exceed `--eval-timeout` receive `--timeout-cost`.

### AdaEva-S: successive halving

Implemented in `_sh_schedule` and the runners in [`src/sh/`](src/sh/).

1. From the pool size $|\mathcal{U}^{(t)}|$, population size $M$ and reduction factor $\eta$ (`--sh-reduction-factor`), the number of rounds is $\lceil \log_\eta(|\mathcal{U}^{(t)}|/M) \rceil$.
2. Instance budgets grow geometrically from `--sh-min-instances` to $K$. The last round always compares the survivors on all $K$ instances.
3. After each round, candidates are ranked by mean cost on the instances they share, and the top $1/\eta$ fraction is promoted.
4. Evaluations are cached, as in AdaEva-R. A budget guard stops a round that would exceed $B$.

### In-loop selection vs. passive incumbent logging

- **Search-time selection** uses only the survivors $\mathcal{V}^{(t)}$ and compares them on the instances they share. This prevents a candidate with a short, lucky prefix from entering $\Pi^{(t+1)}$.
- **Incumbent logging** is passive: it costs no budget and does not feed back into the search. Each generation, `select_incumbent` records $\hat h^{(t)}$ under two rules:
  - `mean_rank`: the best-ranked survivor of the race.
  - `mean_cost`: the lowest mean cost on the instances common to all eligible survivors.
- **Held-out validation.** After the run, both incumbent trajectories are re-evaluated on a disjoint validation or test set (`valid_trajectory_mean_rank.json` / `valid_trajectory_mean_cost.json`). The external benchmarks (TSPLIB, Taillard, held-out BBOB) are evaluated with the scripts in [`src/analyses/`](src/analyses/).

---

## Repository structure

```
AdaEva/
├── src/
│   ├── racing/            # AdaEva-R runners
│   │   ├── base.py        #   RacingBase: generation loop, budget accounting, incumbent logging
│   │   ├── eoh_{obp,tsp_gls,fssp_gls}.py
│   │   ├── hifo_{obp,tsp_gls,fssp_gls}.py
│   │   └── llamea_bbob.py
│   ├── sh/                # AdaEva-S runners (same task grid; hifo_common.py = shared HiFo glue)
│   ├── fixed/             # Uniform-k baseline (--K, --instance-mode fixed|random)
│   ├── full/              # Uniform-K baseline (full evaluation)
│   ├── init_pop/          # Fixed initial populations used with --fix-init-pop
│   ├── analyses/          # Held-out tests, validation re-evaluation, ranking & statistics analyses
│   └── utils/
│       ├── race.py        # elitist_race (AdaEva-R core)
│       ├── tests.py       # Friedman + Conover/Nemenyi, paired t-test
│       ├── config.py      # Config / RaceState: per-(candidate, instance) cost cache
│       ├── ranking.py     # rank matrices with tied mid-ranks
│       ├── llm.py         # OpenAI-compatible / Gemini / Mistral / vLLM / Ollama clients + on-disk cache
│       ├── logger.py      # run directories, stdout mirroring, optional W&B
│       ├── manuscript_log.py, race_log.py, hifo_candidate_log.py   # structured JSONL logs
│       ├── fixed_init_pop.py, source_key.py, obp_utils.py
│       └── updated_tsp_eval/   # Adopted from EoH (examples/tsp_gls_numba): TSP data (TSP20/TSPAEL64/TSPLIB) + GLS evaluator
├── patches/               # Changes applied on top of the upstream packages (see below)
├── scripts/               # Cluster launch scripts (wrappers/, tasks/, submits/)
├── packages/              # Upstream repositories, cloned by the user (git-ignored)
├── .env.example
└── requirements.txt
```


---

## Installation

### 1. Python environment

Python 3.10 is used for all experiments.

```bash
conda create -n adaeva python=3.10 -y
conda activate adaeva
pip install -r requirements.txt
```

`pyconcorde` (the exact TSP solver used for optimality gaps) is built from source and needs a C compiler. If it fails to build, the rest of the code still installs. Only the scripts that compute TSP optimality gaps need it.

### 2. Upstream packages (`packages/`)

AdaEva does not redistribute the host frameworks. Clone five public repositories from their **original** GitHub URLs into `packages/` at the repository root. The code looks them up as `<repo>/packages/<Name>`, so keep the directory names unchanged.

| Directory | Original repository | Pinned commit | Used for |
| :--- | :--- | :--- | :--- |
| `packages/LLM4AD` | https://github.com/Optima-CityU/LLM4AD | `e8d848f` | EoH method; OBP, TSP-GLS and FSSP-GLS tasks |
| `packages/HiFo-Prompt` | https://github.com/Challenger-XJTU/HiFo-Prompt | `e64ce9e` | HiFo-Prompt method |
| `packages/LLaMEA` | https://github.com/XAI-liacs/LLaMEA | `1b86ae6` | LLaMEA method; BBOB utilities |
| `packages/EoH` | https://github.com/FeiLiu36/EoH | `4725457` | FSSP training instances |
| `packages/EoH-S` | https://github.com/FeiLiu36/EoH-S | `8310b05` | Heterogeneous OBP dataset (analyses only) |

**Step 2.1: clone and pin.** Run from the AdaEva repository root:

```bash
mkdir -p packages && cd packages
git clone https://github.com/Optima-CityU/LLM4AD.git       && git -C LLM4AD      checkout e8d848fea67db0626a1669718685afbe338547b2
git clone https://github.com/Challenger-XJTU/HiFo-Prompt.git && git -C HiFo-Prompt checkout e64ce9edbfb4c8ebffd652b785b0c87261785586
git clone https://github.com/XAI-liacs/LLaMEA.git          && git -C LLaMEA      checkout 1b86ae67502c0fb22235175e1005fef9d752d39e
git clone https://github.com/FeiLiu36/EoH.git              && git -C EoH         checkout 472545785c936dcfc863d2bc0d6109cf23c7ce62
git clone https://github.com/FeiLiu36/EoH-S.git            && git -C EoH-S       checkout 8310b056ab0d4ea83d194f1d9e8c7873c0a3b892
cd ..
```

**Step 2.2: apply the AdaEva patches.** Four of the repositories need small changes, which are shipped in [`patches/`](patches/). Run from the repository root:

```bash
for p in LLM4AD HiFo-Prompt LLaMEA EoH-S; do
  git -C packages/$p apply --whitespace=nowarn "$PWD/patches/$p.patch"
done
```

| Patch | Change |
| :--- | :--- |
| `LLM4AD.patch` | Adds the `fssp_gls` task (Numba-accelerated GLS for permutation flow-shop, with the Taillard test sets). Makes `tsp_gls_2O` seedable and able to run without a wall-clock limit. Scales OBP Weibull item sizes with the bin capacity. |
| `HiFo-Prompt.patch` | Moves HiFo's internal constants (insight-pool size, credit weights, stagnation and progress thresholds, timeouts) into a new `hifo_hp.py` so the runners can set them. Fails loudly on an empty population. |
| `LLaMEA.patch` | Makes response parsing robust when no function or class name can be extracted. Fixes stdout redirection. |
| `EoH-S.patch` | Writes the heterogeneous OBP set under the file name used by the analyses. |

**Step 2.3 (optional): generate the heterogeneous OBP dataset.** Generation is seeded, so the result is deterministic. Only `src/analyses/eval_obp_hetero.py` needs this dataset:

```bash
(cd packages/EoH-S/datasets/obp && python generate_instances_weibull_training.py)
# -> packages/EoH-S/datasets/obp/dataset_200_2k_128_5_80.pkl
```

**Step 2.4: expose the packages to Python.** Nothing is `pip install`-ed from `packages/`. Put the sources on `PYTHONPATH` instead (run from the repository root, in every new shell):

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/LLM4AD:$PWD/packages/LLaMEA:$PWD/packages/HiFo-Prompt/hifo/src:$PYTHONPATH"
```

You should end up with:

```
AdaEva/
├── packages/
│   ├── EoH/
│   ├── EoH-S/
│   ├── HiFo-Prompt/
│   ├── LLaMEA/
│   └── LLM4AD/
├── patches/
└── src/
```

`packages/` is git-ignored.

### 3. LLM provider

```bash
cp .env.example .env   # then fill in the key(s) you need
```

| `--llm-backend` | Required variables | Notes |
| :--- | :--- | :--- |
| `openrouter` (default) | `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) | Any OpenAI-compatible endpoint via `OPENAI_BASE_URL`; handles reasoning models (gpt-5\*, o-series) |
| `google` | `GOOGLE_API_KEY` | Gemini via its OpenAI-compatible endpoint; `GOOGLE_THINKING_BUDGET` |
| `mistral` | `MISTRAL_API_KEY` | Client-side throttling via `MISTRAL_MIN_INTERVAL` |
| `vllm` | `VLLM_BASE_URL` | Self-hosted OpenAI-compatible server |
| `ollama` | `OLLAMA_HOST` (or `--ollama-host`) | Local models |

The model is selected with `--llm-model`. The available `--llm-backend` choices differ slightly between runners; see `--help`.

All LLM responses are cached on disk under `.llm_cache/` (override with `--cache-root`). The cache key is a SHA-256 hash of the messages, model and sampling parameters, plus a per-seed salt, so re-running an identical configuration costs no API calls and reproduces the same trajectory.

---

## Running experiments

All runners share the same interface. With `PYTHONPATH` set as above, run each one from the repository root as `python src/<method>/<framework>_<task>.py [options]`, where:

- `<method>` is one of `racing`, `sh`, `fixed` or `full`;
- `<framework>_<task>` is one of `eoh_obp`, `eoh_tsp_gls`, `eoh_fssp_gls`, `hifo_obp`, `hifo_tsp_gls`, `hifo_fssp_gls` or `llamea_bbob`.

### AdaEva-R (racing)

```bash
# EoH on TSP-GLS: Friedman + Conover, budget B = 10 x 20 x 64
python src/racing/eoh_tsp_gls.py \
    --pop-size 10 --ref-max-generations 20 --n-instances 64 --problem-size 100 \
    --t-first 5 --t-each 1 --alpha 0.05 \
    --test-type friedman --posthoc-test-type conover \
    --elitist-limit 2 --fix-init-pop --seed 0 --num-cores 4 \
    --llm-backend google --llm-model gemini-3.1-flash-lite

# LLaMEA on BBOB (24 functions, 5-D)
python src/racing/llamea_bbob.py \
    --n-parents 10 --n-offspring 10 --ref-max-generations 20 --dim 5 \
    --t-first 5 --t-each 2 --posthoc-test-type nemenyi \
    --fix-init-pop --seed 0 --num-cores 4 \
    --llm-backend vllm --llm-model mistralai/Devstral-Small-2-24B-Instruct-2512
```

### AdaEva-S (successive halving)

```bash
python src/sh/eoh_tsp_gls.py \
    --pop-size 10 --ref-max-generations 20 --n-instances 64 \
    --sh-min-instances 5 --sh-reduction-factor 1.33 \
    --fix-init-pop --seed 0 --num-cores 4
```

### Baselines

```bash
# Uniform-k: every candidate on k=4 fixed instances
python src/fixed/eoh_tsp_gls.py --pop-size 10 --max-generations 20 --K 4 --instance-mode fixed --fix-init-pop

# Uniform-K: every candidate on the full training pool
python src/full/eoh_tsp_gls.py --pop-size 10 --max-generations 20 --n-instances 64
```

### Main options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--pop-size` / `--n-parents` | 10 | Population size $M$ |
| `--n-instances` | runner-specific | Training pool size $K$ |
| `--ref-max-generations` | 20 | Reference horizon used to set the default budget $B$ |
| `--budget-cap` | `pop × ref_gens × K` | Evaluation budget $B$ (`-1` disables it) |
| `--max-generations` | runner-specific | Generation cap (`-1` disables it; the run then stops on $B$) |
| `--t-first` / `--t-each` | 5 / 1–2 | Instances before the first test / between tests |
| `--alpha` | 0.05 | Significance level |
| `--test-type` | `friedman` | `friedman` or `ttest` |
| `--posthoc-test-type` | `conover` | `conover` or `nemenyi` |
| `--elitist-limit` | 2 (12 for BBOB) | Stop a race after this many tests that eliminate nothing |
| `--no-elitist` | — | Disable reuse of parents' costs across generations |
| `--save-pop` | off | Refill the population with eliminated candidates when survivors < $M$ |
| `--deterministic` | off | Treat evaluation as deterministic; by default each instance is paired with a seed that is shared by all candidates (irace-style stochastic tasks) |
| `--sh-min-instances` / `--sh-reduction-factor` | 5 / 1.33 | AdaEva-S first-round budget and $\eta$ |
| `--K`, `--instance-mode` | 1, `fixed` | Uniform-$k$ subset size and fixed vs. resampled subset |
| `--fix-init-pop` | off | Start from `src/init_pop/<framework>_<task>.json` |
| `--seed` | 0 | Run seed (instances, LLM cache salt) |
| `--num-cores` / `--num-threads` | 4 | Parallel evaluation workers / LLM sampler threads |
| `--use-wandb` | off | Also log trajectories to Weights & Biases |

Run `python src/<method>/<runner>.py --help` for the full list, including the HiFo-Prompt constants (`--pool-capacity`, `--credit-best`, …) and the LLaMEA options (`--parent-selection`, `--instance-pool-mode`, `--prompt-mode`, …).

### Outputs

Each run writes to `.logs/<label>/<timestamp>_<seed>_<tag>/`:

| File | Content |
| :--- | :--- |
| `trajectory.json`, `heuristics.json` | Per-generation best/incumbent and the code of every heuristic |
| `race_log.jsonl`, `race_step_trace.jsonl`, `race_summary.jsonl` | Race decisions per step and per generation (AdaEva-R/S) |
| `candidate_log.jsonl`, `all_candidates.jsonl` | Candidate lifecycle, genealogy and status (survived / eliminated / crashed / timeout) |
| `population_diversity_log.jsonl`, `fitness_reliability_log.jsonl` | Diversity and partial-vs-full score diagnostics |
| `valid_trajectory_{mean_rank,mean_cost}.json` | Held-out validation of the logged incumbents (`valid_trajectory.json` for the Uniform baselines) |
| `llm_prompts.jsonl`, `timings.json`, `run_meta.json`, `terminal.txt` | LLM I/O and token usage, wall-clock timings, configuration, console log |

---

## Analyses and held-out evaluation

Scripts in [`src/analyses/`](src/analyses/) operate on a run directory (`--exp <run_dir>`):

```bash
# Held-out benchmarks for the final or per-generation incumbent
python src/analyses/test_tsplib.py         --exp .logs/race/tsp_gls/<run> --has-incumbent --incumbent-mode mean_rank
python src/analyses/test_fssp_taillard.py  --exp .logs/race/fssp_gls/<run> --has-incumbent
python src/analyses/test_bbob.py           --exp .logs/racing_llamea_bbob/<run> --has-incumbent
python src/analyses/test_obp_slice.py      --exp .logs/race/obp/<run> --has-incumbent

# Re-evaluate candidates on the full training pool (search-time ranking quality)
python src/analyses/eval_racing_tsp_ranking.py  --exp .logs/race/tsp_gls/<run>
python src/analyses/eval_uniform_tsp_ranking.py --exp .logs/eoh/tiny_tsp_gls/<run>

# Friedman–Conover decision geometry across runs
python src/analyses/statistics_investigation.py --root . --runs .logs/race/tsp_gls/<run> ...
```

- `test_*.py` / `test_*_slice.py`: evaluation on external benchmarks, either of the final heuristic or of checkpoints along the run.
- `eval_{tsp,obp,fssp,bbob}.py`: validation-set re-evaluation, called automatically at the end of a run.
- `eval_{racing,uniform,esh}_*_ranking.py`: partial vs. full-pool ranking agreement.
- `statistics_investigation.py`: concordance → omnibus → post-hoc → elimination chain.

---

## Citation

```bibtex
@inproceedings{anonymous2026adaeva,
  title     = {{AdaEva}: Accelerating {LLM}-Driven Algorithm Design with Adaptive Partial Evaluation},
  author    = {Anonymous},
  booktitle = {Submitted to the International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Under double-blind review}
}
```

## License

MIT. See [LICENSE](LICENSE). The repositories cloned into `packages/` are distributed under their own licenses.

### Third-party code

[`src/utils/updated_tsp_eval/`](src/utils/updated_tsp_eval/) is adopted from the EoH repository, folder [`examples/tsp_gls_numba`](https://github.com/FeiLiu36/EoH/tree/main/examples/tsp_gls_numba) (MIT License, © Fei Liu; Liu et al., *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model*, ICML 2024).
- **Unchanged:** all data files: the TSP20 and TSPAEL64 instance sets and the TSPLIB instances with their optimal tours.
- **Adapted for AdaEva:** `prob.py`, `evaluation/evaluation.py`, `evaluation/heuristic.py` and `evaluation/runEval.py`.
- **Added:** `evaluation/run_eval_eoh.py`.
