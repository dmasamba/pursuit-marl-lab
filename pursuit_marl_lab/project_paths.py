"""Centralized repository paths for scripts and docs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DOCS_ROOT = PROJECT_ROOT / "docs"
ENV_ROOT = PROJECT_ROOT / "envs"
VENDOR_ROOT = PROJECT_ROOT / "vendor"


def artifact_path(*parts: str) -> str:
    return str(ARTIFACTS_ROOT.joinpath(*parts))


def ensure_artifact_dir(*parts: str) -> str:
    path = ARTIFACTS_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
