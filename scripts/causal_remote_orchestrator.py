import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


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
):
    out_path = logs_dir / (
        f"causal_{label}_remote_{model_tag}_n{n}_compactstats_bg_{run_stamp}.out.log"
    )
    err_path = logs_dir / (
        f"causal_{label}_remote_{model_tag}_n{n}_compactstats_bg_{run_stamp}.err.log"
    )
    env = load_env(env_file)
    attempt = 1

    while True:
        cmd = [str(python_exe), "causal_scores.py", "--remote", "--model", model, "--n", str(n)]
        cmd.extend(extra_args)
        if resume_path.exists():
            cmd.append("--resume")
        cmd_str = " ".join(cmd[1:])
        print(f"[{now()}] starting {label} attempt {attempt}: {cmd_str}", flush=True)
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
            print(f"[{now()}] {label} completed successfully", flush=True)
            return 0
        print(
            f"[{now()}] {label} exited with code {rc}; retrying in {retry_delay} seconds",
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
    return parser.parse_args()


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

    concept_resume = (
        repo / "cache" / "causal_scores" / "Llama-3.1-8B" / f"len30_n{args.n}_resume.pkl"
    )
    token_resume = (
        repo
        / "cache"
        / "causal_scores"
        / "Llama-3.1-8B"
        / f"len30_n{args.n}_randoments_resume.pkl"
    )

    with orchestrator_out.open("a", encoding="utf-8") as stdout_log, orchestrator_err.open(
        "a", encoding="utf-8"
    ) as stderr_log:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        try:
            sys.stdout = stdout_log
            sys.stderr = stderr_log
            concept_rc = run_leg(
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
                resume_path=concept_resume,
            )
            if concept_rc != 0:
                return concept_rc
            return run_leg(
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
                resume_path=token_resume,
            )
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


if __name__ == "__main__":
    raise SystemExit(main())
