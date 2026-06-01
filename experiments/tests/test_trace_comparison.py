from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def _run_python_script(args: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=EXPERIMENT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            + " ".join(args)
            + f"\nExit code: {proc.returncode}\n"
            + f"STDOUT (tail):\n{proc.stdout[-4000:]}\n"
            + f"STDERR (tail):\n{proc.stderr[-4000:]}"
        )


def _expdir_for(run_id: str, dgp_name: str) -> Path:
    if dgp_name == "regression-fixed":
        suffix = (
            "name=regression-fixed dim_x=10 noise_std=1.0 "
            "data=8 seed=1001"
        )
    elif dgp_name == "classification-fixed":
        suffix = "name=classification-fixed dim_x=10 data=12 seed=1001"
    else:
        raise ValueError(f"Unsupported dgp_name: {dgp_name}")
    return OUTPUT_DIR / run_id / suffix


@pytest.mark.parametrize(
    ("dgp_name", "data_size"),
    [("regression-fixed", 8), ("classification-fixed", 12)],
)
@pytest.mark.slow
def test_method_runners_from_scratch(dgp_name: str, data_size: int):
    run_id = f"pytest-method-{dgp_name}-{uuid4().hex[:8]}"
    run_dir = OUTPUT_DIR / run_id
    shutil.rmtree(run_dir, ignore_errors=True)

    try:
        _run_python_script(
            [
                "prepare-dgp.py",
                f"id={run_id}",
                f"dgp={dgp_name}",
                f"data_size={data_size}",
                "seed=1001",
            ]
        )
        expdir = _expdir_for(run_id, dgp_name)
        assert (expdir / "configs" / "dgp.yaml").exists()
        assert (expdir / "logs" / "dgp.log").exists()
        assert not (expdir / "dgp.yaml").exists()
        assert not (expdir / ".hydra").exists()
        assert not (expdir / ("setup" + ".yaml")).exists()
        expdir_override = (
            f"expdir='./{expdir.relative_to(EXPERIMENT_DIR).as_posix()}'"
        )

        _run_python_script(
            [
                "run-tabmgp.py",
                expdir_override,
                "seed=1001",
                "rollout_times=2",
                "forward_steps=[3,5]",
                "n_estimators=1",
                "trace=True",
                "resolution=5",
            ]
        )
        _run_python_script(
            [
                "run-bb.py",
                expdir_override,
                "seed=1001",
                "rollout_times=2",
                "forward_steps=[3,5]",
                "trace=True",
                "resolution=5",
            ]
        )
        _run_python_script(
            [
                "run-copula.py",
                expdir_override,
                "seed=1001",
                "rollout_times=2",
                "forward_steps=[3,5]",
                "trace=True",
                "resolution=5",
                "init=std",
            ]
        )
        _run_python_script(
            [
                "run-bayes.py",
                expdir_override,
                "seed=1001",
                "prior=flat",
                "n_warmup=2",
                "n_samples=3",
                "n_chains=1",
            ]
        )
        _run_python_script(
            [
                "run-bb.py",
                expdir_override,
                "run_name=bb-scalar",
                "seed=1001",
                "rollout_times=2",
                "forward_steps=5",
                "trace=False",
            ]
        )
        post_dir = expdir / "posterior-likelihood"
        assert (expdir / "dgp.pickle").exists()
        assert (expdir / "tabmgp-rollout" / "rollout-0.pickle").exists()
        assert not (expdir / "rollout").exists()
        assert not (expdir / ".hydra").exists()

        assert (expdir / "configs" / "tabmgp.yaml").exists()
        assert (expdir / "configs" / "bb.yaml").exists()
        assert (expdir / "configs" / "copula-std.yaml").exists()
        assert (expdir / "configs" / "bayes-flat.yaml").exists()
        assert (expdir / "logs" / "tabmgp.log").exists()
        assert (expdir / "logs" / "bb.log").exists()
        assert (expdir / "logs" / "copula-std.log").exists()
        assert (expdir / "logs" / "bayes-flat.log").exists()

        assert (post_dir / "tabmgp-3-post.pickle").exists()
        assert (post_dir / "tabmgp-5-post.pickle").exists()
        assert not (post_dir / "tabmgp-250-post.pickle").exists()
        assert not list(post_dir.glob("tabpfn-*.pickle"))

        assert (post_dir / "bb-3-post.pickle").exists()
        assert (post_dir / "bb-5-post.pickle").exists()
        assert not (post_dir / "bb-scalar-3-post.pickle").exists()
        assert (post_dir / "bb-scalar-5-post.pickle").exists()
        assert (post_dir / "copula-std-3-post.pickle").exists()
        assert (post_dir / "copula-std-5-post.pickle").exists()
        assert (post_dir / "bayes-flat-post.pickle").exists()
        assert not list(post_dir.glob("gibbs*.pickle"))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
