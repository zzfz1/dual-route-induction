import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def now():
    return datetime.now().astimezone().isoformat()


def load_env(path):
    env = os.environ.copy()
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            env[name.strip()] = value.strip().strip('"').strip("'")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def get_shard_suffix(shard_index, shard_count):
    if shard_count <= 1:
        return ""
    return f"_shard{shard_index}of{shard_count}"


def get_resume_path(repo, model, n, random_tok_entities, shard_index, shard_count):
    model_name = model.split("/")[-1]
    token_suffix = "_randoments" if random_tok_entities else ""
    shard_suffix = get_shard_suffix(shard_index, shard_count)
    return (
        repo
        / "cache"
        / "causal_scores"
        / model_name
        / f"len30_n{n}{token_suffix}{shard_suffix}_resume.pkl"
    )


def run_leg(
    *,
    label,
    python_exe,
    scripts_dir,
    logs_dir,
    model,
    model_tag,
    n,
    run_stamp,
    retry_delay,
    env_file,
    extra_args,
    resume_path,
    remote_max_retries,
    remote_backoff_base,
    remote_backoff_max,
    work_shard_index=0,
    work_shard_count=1,
):
    shard_suffix = get_shard_suffix(work_shard_index, work_shard_count)
    out_path = logs_dir / (
        f"causal_{label}_remote_{model_tag}_n{n}_compactstats_bg_{run_stamp}{shard_suffix}.out.log"
    )
    err_path = logs_dir / (
        f"causal_{label}_remote_{model_tag}_n{n}_compactstats_bg_{run_stamp}{shard_suffix}.err.log"
    )
    env = load_env(env_file)
    attempt = 1

    while True:
        cmd = [
            str(python_exe),
            "causal_scores.py",
            "--remote",
            "--model",
            model,
            "--n",
            str(n),
            "--remote-max-retries",
            str(remote_max_retries),
            "--remote-backoff-base",
            str(remote_backoff_base),
            "--remote-backoff-max",
            str(remote_backoff_max),
            "--work-shard-index",
            str(work_shard_index),
            "--work-shard-count",
            str(work_shard_count),
        ]
        cmd.extend(extra_args)
        if resume_path.exists():
            cmd.append("--resume")
        cmd_str = " ".join(cmd[1:])
        shard_label = (
            f"{label}{shard_suffix}" if shard_suffix else label
        )
        print(
            f"[{now()}] starting {shard_label} attempt {attempt}: {cmd_str}",
            flush=True,
        )
        with out_path.open("a", encoding="utf-8") as fout, err_path.open(
            "a", encoding="utf-8"
        ) as ferr:
            proc = subprocess.Popen(
                cmd,
                cwd=scripts_dir,
                stdout=fout,
                stderr=ferr,
                env=env,
            )
            rc = proc.wait()
        if rc == 0:
            print(f"[{now()}] {shard_label} completed successfully", flush=True)
            return 0
        print(
            f"[{now()}] {shard_label} exited with code {rc}; retrying in {retry_delay} seconds",
            flush=True,
        )
        attempt += 1
        time.sleep(retry_delay)


def run_helper_command(
    *,
    helper_label,
    python_exe,
    scripts_dir,
    env_file,
    retry_delay,
    cmd_args,
):
    env = load_env(env_file)
    attempt = 1
    while True:
        cmd = [str(python_exe), "causal_scores.py", *cmd_args]
        print(
            f"[{now()}] starting {helper_label} attempt {attempt}: {' '.join(cmd[1:])}",
            flush=True,
        )
        proc = subprocess.Popen(cmd, cwd=scripts_dir, env=env)
        rc = proc.wait()
        if rc == 0:
            print(f"[{now()}] {helper_label} completed successfully", flush=True)
            return 0
        print(
            f"[{now()}] {helper_label} exited with code {rc}; retrying in {retry_delay} seconds",
            flush=True,
        )
        attempt += 1
        time.sleep(retry_delay)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrying orchestrator for NDIF remote causal concept/token runs."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        help="Model name to pass to causal_scores.py.",
    )
    parser.add_argument(
        "--model-tag",
        default="llama31",
        help="Short tag used in log filenames.",
    )
    parser.add_argument("--n", type=int, default=1024, help="Number of examples.")
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=60,
        help="Seconds to wait before retrying a failed leg.",
    )
    parser.add_argument(
        "--run-stamp",
        default=None,
        help="Optional timestamp override for log filenames.",
    )
    parser.add_argument(
        "--parallel-legs",
        action="store_true",
        help="Run concept and token legs concurrently instead of serially.",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=1,
        help="Number of shard processes to run per leg.",
    )
    parser.add_argument(
        "--remote-max-retries",
        type=int,
        default=4,
        help="Maximum retries for a single NDIF request before failing the leg.",
    )
    parser.add_argument(
        "--remote-backoff-base",
        type=float,
        default=2.0,
        help="Initial backoff in seconds for retryable NDIF request failures.",
    )
    parser.add_argument(
        "--remote-backoff-max",
        type=float,
        default=30.0,
        help="Maximum backoff in seconds for retryable NDIF request failures.",
    )
    return parser.parse_args()


def run_sharded_leg(
    *,
    repo,
    label,
    python_exe,
    scripts_dir,
    logs_dir,
    model,
    model_tag,
    n,
    run_stamp,
    retry_delay,
    env_file,
    extra_args,
    process_workers,
    remote_max_retries,
    remote_backoff_base,
    remote_backoff_max,
):
    if process_workers <= 1:
        resume_path = get_resume_path(
            repo, model, n, "--random_tok_entities" in extra_args, 0, 1
        )
        return run_leg(
            label=label,
            python_exe=python_exe,
            scripts_dir=scripts_dir,
            logs_dir=logs_dir,
            model=model,
            model_tag=model_tag,
            n=n,
            run_stamp=run_stamp,
            retry_delay=retry_delay,
            env_file=env_file,
            extra_args=extra_args,
            resume_path=resume_path,
            remote_max_retries=remote_max_retries,
            remote_backoff_base=remote_backoff_base,
            remote_backoff_max=remote_backoff_max,
        )

    random_tok_entities = "--random_tok_entities" in extra_args
    prep_rc = run_helper_command(
        helper_label=f"{label}_prepare",
        python_exe=python_exe,
        scripts_dir=scripts_dir,
        env_file=env_file,
        retry_delay=retry_delay,
        cmd_args=[
            "--remote",
            "--model",
            model,
            "--n",
            str(n),
            "--prepare-work-items",
            "--work-shard-count",
            str(process_workers),
            *(["--random_tok_entities"] if random_tok_entities else []),
        ],
    )
    if prep_rc != 0:
        return prep_rc

    shard_kwargs = []
    for shard_idx in range(process_workers):
        shard_kwargs.append(
            dict(
                label=label,
                python_exe=python_exe,
                scripts_dir=scripts_dir,
                logs_dir=logs_dir,
                model=model,
                model_tag=model_tag,
                n=n,
                run_stamp=run_stamp,
                retry_delay=retry_delay,
                env_file=env_file,
                extra_args=extra_args,
                resume_path=get_resume_path(
                    repo,
                    model,
                    n,
                    random_tok_entities,
                    shard_idx,
                    process_workers,
                ),
                remote_max_retries=remote_max_retries,
                remote_backoff_base=remote_backoff_base,
                remote_backoff_max=remote_backoff_max,
                work_shard_index=shard_idx,
                work_shard_count=process_workers,
            )
        )

    with ThreadPoolExecutor(max_workers=process_workers) as executor:
        shard_futures = [executor.submit(run_leg, **kwargs) for kwargs in shard_kwargs]
        shard_rcs = [future.result() for future in shard_futures]
    if any(rc != 0 for rc in shard_rcs):
        return next(rc for rc in shard_rcs if rc != 0)

    return run_helper_command(
        helper_label=f"{label}_merge",
        python_exe=python_exe,
        scripts_dir=scripts_dir,
        env_file=env_file,
        retry_delay=retry_delay,
        cmd_args=[
            "--remote",
            "--model",
            model,
            "--n",
            str(n),
            "--merge-shards",
            "--work-shard-count",
            str(process_workers),
            *(["--random_tok_entities"] if random_tok_entities else []),
        ],
    )


def main():
    args = parse_args()

    scripts_dir = Path(__file__).resolve().parent
    repo = scripts_dir.parent
    project = repo.parent
    logs_dir = repo / "logs" / "ndif"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_file = project / ".env"
    python_exe = Path(sys.executable)
    run_stamp = args.run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    orchestrator_out = logs_dir / (
        f"causal_remote_{args.model_tag}_n{args.n}_compactstats_orchestrator_{run_stamp}.out.log"
    )
    orchestrator_err = logs_dir / (
        f"causal_remote_{args.model_tag}_n{args.n}_compactstats_orchestrator_{run_stamp}.err.log"
    )

    with orchestrator_out.open("a", encoding="utf-8") as stdout_log, orchestrator_err.open(
        "a", encoding="utf-8"
    ) as stderr_log:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        try:
            sys.stdout = stdout_log
            sys.stderr = stderr_log
            concept_kwargs = dict(
                label="concept",
                python_exe=python_exe,
                scripts_dir=scripts_dir,
                logs_dir=logs_dir,
                model=args.model,
                model_tag=args.model_tag,
                n=args.n,
                run_stamp=run_stamp,
                retry_delay=args.retry_delay,
                env_file=env_file,
                extra_args=[],
                process_workers=args.process_workers,
                remote_max_retries=args.remote_max_retries,
                remote_backoff_base=args.remote_backoff_base,
                remote_backoff_max=args.remote_backoff_max,
            )
            token_kwargs = dict(
                label="token",
                python_exe=python_exe,
                scripts_dir=scripts_dir,
                logs_dir=logs_dir,
                model=args.model,
                model_tag=args.model_tag,
                n=args.n,
                run_stamp=run_stamp,
                retry_delay=args.retry_delay,
                env_file=env_file,
                extra_args=["--random_tok_entities"],
                process_workers=args.process_workers,
                remote_max_retries=args.remote_max_retries,
                remote_backoff_base=args.remote_backoff_base,
                remote_backoff_max=args.remote_backoff_max,
            )

            if args.parallel_legs:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    concept_future = executor.submit(run_sharded_leg, repo=repo, **concept_kwargs)
                    token_future = executor.submit(run_sharded_leg, repo=repo, **token_kwargs)
                    concept_rc = concept_future.result()
                    token_rc = token_future.result()
                return concept_rc or token_rc

            concept_rc = run_sharded_leg(repo=repo, **concept_kwargs)
            if concept_rc != 0:
                return concept_rc
            return run_sharded_leg(repo=repo, **token_kwargs)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


if __name__ == "__main__":
    raise SystemExit(main())
