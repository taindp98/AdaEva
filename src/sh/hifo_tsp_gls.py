"""Successive-Halving HiFo-Prompt on the LLM4AD TSP-GLS task.

The SH counterpart of ``racing/hifo_tsp_gls.py`` (and the HiFo counterpart of
``sh/eoh_tsp_gls.py``). It reuses ``racing/hifo_tsp_gls``'s HiFo-Prompt candidate
generation (InsightPool + EvolutionaryNavigator + ``m3``, nested per-operator
sub-races) VERBATIM and only swaps the Friedman/Nemenyi ``elitist_race`` for an elitist
**Successive-Halving** sub-race via ``SHHiFo(SHHiFoRaceMixin, RacingHiFo)`` (see
``sh.hifo_common``). Each of the five per-operator sub-races becomes an SH ladder.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pathlib
import random
import sys
import time
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "packages" / "HiFo-Prompt" / "hifo" / "src"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, OpenRouterClient, GoogleClient, OllamaClient,
                   MistralClient, vLLMClient, make_wandb_logger)
from utils.llm import OpenRouterLLM4AD, OllamaLLM4AD, MistralLLM4AD, vLLMLLM4AD, GoogleLLM4AD
from utils.logger import make_log_dir
from llm4ad.task.optimization.tsp_gls_2O.evaluation import TSP_GLS_2O_Evaluation_wo_Time
from llm4ad.task.optimization.tsp_gls_2O.get_instance import GetData

from racing.base import _Tee, _run_tag
from racing.eoh_tsp_gls import score_tsp_inst, _opt_cost, BIG_PENALTY
from racing.hifo_tsp_gls import (RacingHiFo, _inject_hifo_hp, _OPERATORS, _N_INIT_BATCHES,
                                 _BASH_DEFAULTS as _RACE_BASH_DEFAULTS,
                                 _ABBREV as _RACE_ABBREV)
from sh.hifo_common import SHHiFoRaceMixin

_BASH_DEFAULTS: dict = {**_RACE_BASH_DEFAULTS,
                        "sh_reduction_factor": 1.25, "sh_min_instances": 5}
_ABBREV: dict = {**_RACE_ABBREV,
                 "sh_reduction_factor": "srf", "sh_min_instances": "smi"}


class SHHiFo(SHHiFoRaceMixin, RacingHiFo):
    """``RacingHiFo`` with a Successive-Halving sub-race (see ``sh.hifo_common``)."""

    def _run_header(self) -> str:
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        return (f"[{self.label}] elitist successive halving HiFo TSP-GLS "
                f"(budget_cap={cap_str}, max_generations={gen_str}, "
                f"eta={self.sh_reduction_factor:g}, N_min={self.sh_min_instances}, "
                f"N_target={self.pop_size})")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Successive-Halving HiFo-Prompt on TSP-GLS.")
    p.add_argument("--pop-size", type=int, default=10, help="HiFo ec_pop_size (fix-init loads 2*pop_size).")
    p.add_argument("--max-generations", type=int, default=20, help="Generation cap. Use -1 to disable.")
    p.add_argument("--ref-max-generations", type=int, default=20,
                   help="Reference HiFo generations for the budget cap when --budget-cap "
                        "is unset (matching reprod/hifo_tsp_gls's total evaluations).")
    p.add_argument("--n-instances", type=int, default=64)
    p.add_argument("--problem-size", type=int, default=100, help="Number of nodes per TSP instance.")
    p.add_argument("--eval-timeout", type=float, default=65.0,
                   help="Per-(candidate, instance) GLS wall-clock cap (s).")
    p.add_argument("--timeout-cost", type=float, default=BIG_PENALTY,
                   help=f"Finite penalty for a heuristic killed at --eval-timeout. Default {BIG_PENALTY:g}.")
    p.add_argument("--selection-num", type=int, default=5)
    p.add_argument("--t-first", type=int, default=5)
    p.add_argument("--t-each", type=int, default=1)
    p.add_argument("--sh-reduction-factor", type=float, default=1.25,
                   help="SH reduction factor eta (each round prunes ~floor(K/eta)).")
    p.add_argument("--sh-min-instances", type=int, default=5,
                   help="Instances in the FIRST SH round (N_min); grows to the full raced set.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--test-type", type=str, default="friedman", choices=["friedman", "ttest"])
    p.add_argument("--posthoc-test-type", type=str, default="conover", choices=["conover", "nemenyi"])
    p.add_argument("--no-elitist", dest="elitist", action="store_false", default=True)
    p.add_argument("--elitist-new-instances", type=int, default=1)
    p.add_argument("--elitist-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-cap", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--fitness-mode", "--prompt-mode", dest="fitness_mode", type=str,
                   default="partial_eval", choices=["original", "partial_eval"],
                   help="Scalar driving HiFo's Hindsight/Foresight/parent-selection "
                        "(does NOT change the prompt text or the SH race).")
    p.add_argument("--label", type=str, default="sh/hifo_tsp_gls")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048)
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"])
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next")
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--num-cores", type=int, default=1)
    p.add_argument("--save-pop", action="store_true", default=False)
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed generation 0 from src/init_pop/hifo_tsp_gls.json (2*pop_size).")
    p.add_argument("--early-stopping-non-elitist", action="store_true", default=False)
    p.add_argument("--deal-with-crashed", type=str, default="rejection", choices=["rejection", "penalty"])
    p.add_argument("--use-wandb", action="store_true", default=False)

    g_ip = p.add_argument_group("HiFo Insight Pool (Hindsight)")
    g_ip.add_argument("--pool-capacity", type=int, default=30)
    g_ip.add_argument("--novelty-threshold", type=float, default=0.7)
    g_ip.add_argument("--selection-count", type=int, default=3)
    g_ip.add_argument("--usage-penalty-weight", type=float, default=0.1)
    g_ip.add_argument("--recency-bonus", type=float, default=0.2)
    g_ip.add_argument("--recency-window", type=int, default=2)
    g_ip.add_argument("--ema-alpha", type=float, default=0.3)
    g_ip.add_argument("--decay-rate", type=float, default=0.01)
    g_ip.add_argument("--probation-usage", type=int, default=3)
    g_cr = p.add_argument_group("HiFo credit-assignment tiers (Eq. 3 intercepts)")
    g_cr.add_argument("--credit-best", type=float, default=0.8)
    g_cr.add_argument("--credit-inc", type=float, default=0.2)
    g_cr.add_argument("--credit-pen", type=float, default=-0.3)
    g_nav = p.add_argument_group("HiFo Evolutionary Navigator (Foresight)")
    g_nav.add_argument("--progress-eps", type=float, default=1e-4)
    g_nav.add_argument("--stagnation-threshold", type=int, default=3)
    g_nav.add_argument("--progress-threshold", type=int, default=2)
    g_nav.add_argument("--diversity-threshold", type=float, default=0.3)
    return p.parse_args(argv)


def main(argv=None) -> int:
    main_t0 = time.time()
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)

    now = _dt.datetime.now()
    dt_stamp = args.run_stamp or f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"
    log_dir = make_log_dir(args.log_root, args.label, dt_stamp, args.seed,
                           tag=_run_tag(args, _BASH_DEFAULTS, _ABBREV))
    cache_dir = args.cache_root / dt_stamp / str(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import atexit
    term_log_path = log_dir / "terminal.txt"
    term_log_file = open(term_log_path, "w", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, term_log_file)
    sys.stderr = _Tee(_orig_stderr, term_log_file)

    def _restore_streams():
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        term_log_file.close()

    atexit.register(_restore_streams)
    print(f"terminal output mirrored -> {term_log_path}")

    max_generations = (None if args.max_generations is not None and args.max_generations < 0
                       else args.max_generations)
    if args.budget_cap is None:
        cand_evals = (_N_INIT_BATCHES + args.ref_max_generations * len(_OPERATORS)) * args.pop_size
        budget_cap = cand_evals * args.n_instances
        print(f"budget_cap {budget_cap} = ({_N_INIT_BATCHES} init + "
              f"{args.ref_max_generations} HiFo-gen x {len(_OPERATORS)} ops) x "
              f"{args.pop_size} pop x {args.n_instances} instances "
              f"(aligned with reprod/hifo_tsp_gls total evaluations)")
    else:
        budget_cap = None if args.budget_cap < 0 else args.budget_cap
    if max_generations is None and budget_cap is None:
        print("error: at least one of --max-generations or --budget-cap must be >= 0", file=sys.stderr)
        return 2

    print(f"pop_size={args.pop_size}  "
          f"max_generations={'off' if max_generations is None else max_generations}  "
          f"n_instances={args.n_instances}  problem_size={args.problem_size}")
    print(f"offspring/gen = {len(_OPERATORS)} operators x {args.pop_size} = "
          f"{len(_OPERATORS) * args.pop_size} (NESTED: per-operator SH sub-races)   "
          f"fitness_mode={args.fitness_mode}")
    print(f"SH schedule: eta={args.sh_reduction_factor:g}, N_min={args.sh_min_instances}")
    print(f"budget_cap={'off' if budget_cap is None else budget_cap}")

    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir, prompt_log=log_dir / "llm_prompts.jsonl"); llm = OllamaLLM4AD(cached)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir, prompt_log=log_dir / "llm_prompts.jsonl"); llm = MistralLLM4AD(cached)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
        cached = CachedLLM(client, cache_dir=cache_dir, prompt_log=log_dir / "llm_prompts.jsonl"); llm = vLLMLLM4AD(cached)
    elif args.llm_backend == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            print("GOOGLE_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = GoogleClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
        cached = CachedLLM(client, cache_dir=cache_dir, prompt_log=log_dir / "llm_prompts.jsonl"); llm = GoogleLLM4AD(cached)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
        cached = CachedLLM(client, cache_dir=cache_dir, prompt_log=log_dir / "llm_prompts.jsonl"); llm = OpenRouterLLM4AD(cached)
    print(f"Backend: {args.llm_backend}  model: {cached.client.model}  cache: {cache_dir}")

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    args_dict["llm_model"] = cached.client.model
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args, _BASH_DEFAULTS, _ABBREV)}"
    from utils.manuscript_log import write_run_meta as _wrm
    _wrm(log_dir, {
        "run_id": run_name, "framework": "HiFo-Prompt", "domain": "TSP-GLS", "policy": "SH",
        "seed": args.seed, "M": args.pop_size, "N": args.n_instances, "K": args.n_instances,
        "alpha": args.alpha, "test_type": args.test_type, "posthoc": args.posthoc_test_type,
        "T_first": args.t_first, "T_each": args.t_each, "elimit": args.elitist_limit,
        "refill": bool(args.save_pop), "incumbent_rule": "both",
        "B": budget_cap, "llm_model": cached.client.model,
    })
    wandb_logger = make_wandb_logger(enabled=args.use_wandb, project="llm4ad",
                                     name=run_name, config=args_dict)

    _inject_hifo_hp(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluation = TSP_GLS_2O_Evaluation_wo_Time()
    evaluation.n_instance = args.n_instances
    evaluation.problem_size = args.problem_size
    if (args.n_instances, args.problem_size) != (16, 100):
        evaluation._datasets = GetData(args.n_instances, args.problem_size).generate_instances()
    random.seed(args.seed)
    np.random.seed(args.seed)

    instances = list(evaluation._datasets)
    for i, inst in enumerate(instances):
        inst._id = i
    print(f"  instances: n={args.n_instances}  problem_size={args.problem_size}  "
          f"eval_timeout={args.eval_timeout}s/instance")

    print("Solving per-instance optima with Concorde...")
    opt_per_inst = [_opt_cost(inst) for inst in instances]
    mean_opt = float(np.mean(opt_per_inst))
    opt_by_idx = {i: v for i, v in enumerate(opt_per_inst)}
    print(f"  per-instance opt: min={min(opt_per_inst):.4f}  "
          f"max={max(opt_per_inst):.4f}  mean={mean_opt:.4f}")

    _racer = SHHiFo(
        evaluation=evaluation, score_one=score_tsp_inst, instances=instances,
        label=args.label, log_dir=log_dir, llm=llm, pop_size=args.pop_size,
        max_generations=max_generations, selection_num=args.selection_num,
        budget_cap=budget_cap, t_first=args.t_first, t_each=args.t_each, alpha=args.alpha,
        seed=args.seed, opt_by_idx=opt_by_idx, mean_opt=mean_opt, problem_size=args.problem_size,
        num_threads=args.num_threads, num_cores=args.num_cores,
        eval_timeout=args.eval_timeout, timeout_cost=args.timeout_cost,
        test_type=args.test_type, posthoc_test_type=args.posthoc_test_type,
        elitist=args.elitist, elitist_new_instances=args.elitist_new_instances,
        elitist_limit=args.elitist_limit, save_pop=args.save_pop,
        early_stopping_non_elitist=args.early_stopping_non_elitist,
        deal_with_crashed=args.deal_with_crashed,
        deterministic=args.deterministic, fitness_mode=args.fitness_mode,
        sh_reduction_factor=args.sh_reduction_factor,
        sh_min_instances=args.sh_min_instances,
        wandb_logger=wandb_logger,
    )
    if args.fix_init_pop:
        from utils.fixed_init_pop import load_fixed_initial_population
        _fp = ROOT / "src" / "init_pop" / "hifo_tsp_gls.json"
        _racer._fixed_init_sources = [
            h["source"] for h in load_fixed_initial_population(_fp, 2 * _racer.pop_size)]
        print(f"  [fixed-init-pop] loaded {len(_racer._fixed_init_sources)} "
              f"heuristics from {_fp}", flush=True)
    _racer.run()

    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
