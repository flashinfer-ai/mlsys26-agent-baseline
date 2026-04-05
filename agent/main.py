import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from agent.api import create_inference_server  # noqa: E402
from agent.eval import create_eval_fn, read_metrics  # noqa: E402
from agent.evolve_agent import run_evolve_loop  # noqa: E402
from agent.iterative_agent import run_iterative_loop  # noqa: E402
from agent.utils import load_config_from_yaml  # noqa: E402
from agent.utils import (
    REPO_TOP_PATH,
    get_dataset_root,
    load_tasks_from_test_list,
    load_test_source,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_agent(args: argparse.Namespace, inference_server, level, problem_id):
    """
    Run agent on a single problem.
    For FIT/MLSYS: level is op_type name (str), problem_id is problem name (str)
    """
    result_save_path = os.path.join(args.save_path, f"{level}_{problem_id}")
    os.makedirs(result_save_path, exist_ok=True)
    problem_name, ref_arch_src = load_test_source(args.test_source, level, problem_id)

    args.level = level
    args.problem_id = problem_id

    args.task_params = {
        "definition": json.dumps(ref_arch_src, indent=4),
        "target_gpu": args.gpu_name,
        "gpu_name": args.gpu_name,
        "gpu_architecture": args.gpu_architecture,
        "dtype_str": str(ref_arch_src.get("inputs", "unknown")),
    }

    with open(os.path.join(result_save_path, "reference_src.py"), "w") as f:
        f.write(json.dumps(ref_arch_src, indent=4))

    if args.agent_type == "iterative":
        args.refine_steps = args.total_steps
        best_kernel, best_metrics = run_iterative_loop(
            ref_arch_src, inference_server, None, None, args, log_path=result_save_path
        )
    elif args.agent_type == "evolve":
        args.proposal_steps = args.total_steps
        args.refine_steps = 0
        best_kernel, best_metrics = run_evolve_loop(
            ref_arch_src, inference_server, args, log_path=result_save_path
        )
    else:
        raise ValueError(f"Invalid agent type: {args.agent_type}")

    with open(
        os.path.join(result_save_path, f"global_best_kernel_{args.total_steps}.py"), "w"
    ) as f:
        f.write(best_kernel)
    with open(
        os.path.join(result_save_path, f"global_best_metrics_{args.total_steps}.json"),
        "w",
    ) as f:
        json.dump(best_metrics.model_dump(), f, indent=4)

    return best_kernel, best_metrics


def _check_cached_result(args, level, problem_id):
    """Return cached (correctness, speedup) if result already exists, else None."""
    result_save_path = os.path.join(args.save_path, f"{level}_{problem_id}")
    if not os.path.exists(result_save_path):
        return None

    kernel_file = os.path.join(
        result_save_path, f"global_best_kernel_{args.total_steps}.py"
    )
    metrics_file = os.path.join(
        result_save_path, f"global_best_metrics_{args.total_steps}.json"
    )

    if os.path.exists(kernel_file) and os.path.exists(metrics_file):
        return read_metrics(metrics_file)

    # Incomplete result, clean up (unless resuming from it)
    if not (args.resume_from and args.resume_from == result_save_path):
        shutil.rmtree(result_save_path)
    return None


def run_main_loop(args):
    """Run the agent on all tasks and report results."""
    tasks = load_tasks_from_test_list(args.tasks_path, args.test_source)
    inference_server = create_inference_server(api_type=args.api_type)
    resume_root = args.resume_from

    correct_count = 0
    sum_speedup = 0.0
    total_count = len(tasks)

    for task in tqdm(tasks, desc="Processing problems"):
        level, problem_id = task["level"], task["problem_id"]

        # Handle resume path
        if resume_root:
            resume_path = os.path.join(resume_root, f"{level}_{problem_id}")
            args.resume_from = resume_path if os.path.exists(resume_path) else None
        else:
            args.resume_from = None

        # Skip already-completed problems
        cached = _check_cached_result(args, level, problem_id)
        if cached is not None:
            correctness, speedup = cached
            if correctness:
                correct_count += 1
                sum_speedup += speedup
            print(
                f"Completed: level {level} problem {problem_id} - "
                f"Correct: {correctness}, Speedup: {speedup:.4f}"
            )
            continue

        try:
            best_kernel, best_metrics = run_agent(
                args, inference_server, level, problem_id
            )
            if best_metrics.correct:
                correct_count += 1
                sum_speedup += best_metrics.speedup
            print(
                f"Completed: level {level} problem {problem_id} - "
                f"Correct: {best_metrics.correct}, "
                f"Speedup: {best_metrics.speedup:.4f}"
            )
        except Exception as e:
            print(f"Failed: level {level} problem {problem_id} - {e}")
            traceback.print_exc()

    print(
        f"Correct count: {correct_count}, Sum speedup: {sum_speedup}, Total count: {total_count}"
    )
    if total_count > 0:
        print(f"Average speedup: {sum_speedup / total_count}")
        print(f"Accuracy: {correct_count / total_count}")
    else:
        print("Average speedup: N/A (no test problems)")
        print("Accuracy: N/A (no test problems)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Test Configs
    parser.add_argument(
        "--test_source",
        type=str,
        default="mlsys26-contest",
        choices=["mlsys26-contest", "flashinfer-trace"],
    )
    parser.add_argument(
        "--agent_type", type=str, default="iterative", choices=["iterative", "evolve"]
    )
    parser.add_argument("--tasks_path", type=str, default=None)
    parser.add_argument("--dtype_str", type=str, default="fp32")
    parser.add_argument("--gpu_name", type=str, default="A6000")
    parser.add_argument("--gpu_architecture", type=str, default="Ampere")

    # Base Model Configs
    parser.add_argument(
        "--api_type", type=str, default="openai", choices=["openai", "claude", "gemini"]
    )
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)

    # Resume Configs
    parser.add_argument("--resume_from", type=str, default=None)

    # Search Configs
    parser.add_argument("--total_steps", type=int, default=25)
    parser.add_argument("--max_memory_round", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--softmax_temperature", type=float, default=1.0)

    # Eval Backend Configs
    parser.add_argument(
        "--eval_backend",
        type=str,
        default="local",
        choices=["local", "modal"],
    )
    parser.add_argument("--modal_gpu", type=str, default="B200")

    # Logging Configs
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    # Config file
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    args = load_config_from_yaml(args, parser)
    start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    if args.save_path is None:
        args.save_path = os.path.join(
            REPO_TOP_PATH,
            "outputs",
            f"{args.agent_type}_{args.test_source}_{args.total_steps}_{start_time}",
        )

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    if args.eval_backend == "modal":
        import modal

        from agent.modal_eval import create_modal_app, ensure_dataset_synced

        modal_app, remote_eval_fn, dataset_vol = create_modal_app(args.modal_gpu)
        args.eval_fn = create_eval_fn(
            "modal", args.test_source, remote_fn=remote_eval_fn
        )
        with modal.enable_output(), modal_app.run():
            ensure_dataset_synced(
                dataset_vol, get_dataset_root(args.test_source), args.test_source
            )
            run_main_loop(args)
    else:
        args.eval_fn = create_eval_fn("local")
        run_main_loop(args)
