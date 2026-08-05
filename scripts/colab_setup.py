"""One-call environment bootstrap for Google Colab.

Usage, as the entire first cell of any notebook::

    !git clone https://github.com/YOUR_USER/GeoQ.git /content/GeoQ 2>/dev/null || \
        (cd /content/GeoQ && git pull)
    %run /content/GeoQ/scripts/colab_setup.py

Why this is a script and not a block of notebook cells
------------------------------------------------------
Setup code pasted into notebooks drifts. Three notebooks acquire three
slightly different versions, one of them pins a different library version,
and eventually two experiments run against different environments while
appearing identical in the manuscript. A version-controlled script has one
definition, and its git hash is recorded alongside every result.

What it guarantees
------------------
* Drive is mounted before any path is resolved, so a disconnect cannot cause a
  silent write to Colab's ephemeral local disk -- which looks like success and
  loses everything on the next restart.
* The package is installed in editable mode, so ``import geoq`` works from any
  working directory without ``sys.path`` manipulation.
* The workspace tree exists under Drive.
* The exact environment is printed and returned, so it can be written into the
  provenance record of the run that follows.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

#  Default Drive workspace. Overridable via the function argument; never
#  hardcoded at a call site.
DEFAULT_WORKSPACE = Path("/content/drive/MyDrive/GeoQ_workspace")

#  Subdirectories created under the workspace. Results, checkpoints and logs
#  live in Drive rather than in the cloned repository, because the repository
#  is disposable on Colab and Drive is not.
WORKSPACE_SUBDIRS = (
    "data",
    "results",
    "checkpoints",
    "logs",
    "figures",
    "tables",
    "artifacts",
)


@dataclass(frozen=True)
class Environment:
    """A snapshot of the runtime, recorded with every experiment.

    Attributes:
        python_version: Interpreter version string.
        platform: Operating system and architecture.
        repo_root: Location of the cloned repository.
        workspace: Drive-backed directory holding all outputs.
        git_commit: Commit hash of the repository, or ``"unknown"``.
        git_dirty: Whether the working tree has uncommitted changes. A dirty
            tree means the commit hash does not describe the code that ran,
            which is the single most common cause of a result that cannot be
            reproduced later.
        accelerator: Detected GPU name, or ``"cpu"``.
        packages: Versions of the scientific libraries that affect results.
    """

    python_version: str
    platform: str
    repo_root: str
    workspace: str
    git_commit: str
    git_dirty: bool
    accelerator: str
    packages: dict[str, str] = field(default_factory=dict)


def _run(command: list[str], cwd: Path | None = None) -> str:
    """Run a command and return its stripped stdout, or an empty string."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def in_colab() -> bool:
    """Return whether the current interpreter is running inside Colab."""
    return "google.colab" in sys.modules or Path("/content").is_dir()


def mount_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive, unless it is already mounted or this is not Colab.

    Args:
        mount_point: Where Drive is mounted.

    Raises:
        RuntimeError: If mounting fails on Colab. This is fatal by design:
            continuing would write results to the ephemeral local disk, where
            they look saved and are destroyed by the next runtime restart.
    """
    if not in_colab():
        print("[setup] Not running on Colab; skipping Drive mount.")
        return
    if (Path(mount_point) / "MyDrive").is_dir():
        print(f"[setup] Drive already mounted at {mount_point}.")
        return
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mount_point)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to mount Google Drive at {mount_point}. Refusing to "
            f"continue: without Drive, every result and checkpoint would be "
            f"written to ephemeral local storage and lost on the next runtime "
            f"restart."
        ) from exc


def install_package(repo_root: Path, extras: str = "all") -> None:
    """Install the repository in editable mode.

    Editable mode matters on Colab: after ``git pull``, the new code is live
    immediately with no reinstall, so a mid-session fix does not cost a
    ten-minute dependency resolution.

    Args:
        repo_root: Path to the repository containing ``pyproject.toml``.
        extras: Optional-dependency group to install.

    Raises:
        FileNotFoundError: If ``pyproject.toml`` is absent.
        RuntimeError: If the installation fails.
    """
    if not (repo_root / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"No pyproject.toml under {repo_root}. Clone the repository first, "
            f"or pass the correct repo_root."
        )
    print(f"[setup] Installing geoq[{extras}] from {repo_root} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f".[{extras}]"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Editable install failed:\n{result.stderr[-2000:]}")
    print("[setup] Install complete.")


def create_workspace(workspace: Path) -> Path:
    """Create the Drive-backed output tree.

    Args:
        workspace: Root of the workspace.

    Returns:
        The workspace path.
    """
    for name in WORKSPACE_SUBDIRS:
        (workspace / name).mkdir(parents=True, exist_ok=True)
    print(f"[setup] Workspace ready at {workspace}")
    return workspace


def detect_accelerator() -> str:
    """Return the GPU name if one is present, otherwise ``"cpu"``."""
    name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return name.splitlines()[0].strip() if name else "cpu"


def collect_versions() -> dict[str, str]:
    """Return installed versions of the libraries that can change results.

    Only libraries whose version can alter a number are listed. Recording the
    version of a plotting library would be noise; recording NumPy's is not,
    because its LAPACK backend affects the last digits of an eigendecomposition
    and therefore of every geodesic distance in the thesis.
    """
    from importlib.metadata import PackageNotFoundError, version

    watched = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "pyriemann",
        "mne",
        "moabb",
        "pennylane",
        "qiskit",
        "qiskit-aer",
        "geoq",
    ]
    versions: dict[str, str] = {}
    for package in watched:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def describe_environment(repo_root: Path, workspace: Path) -> Environment:
    """Build the environment snapshot for the provenance record.

    Args:
        repo_root: Repository location.
        workspace: Drive workspace location.

    Returns:
        A populated :class:`Environment`.
    """
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root) or "unknown"
    dirty = bool(_run(["git", "status", "--porcelain"], cwd=repo_root))
    return Environment(
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        repo_root=str(repo_root),
        workspace=str(workspace),
        git_commit=commit,
        git_dirty=dirty,
        accelerator=detect_accelerator(),
        packages=collect_versions(),
    )


def setup(
    repo_root: str | Path = "/content/GeoQ",
    workspace: str | Path = DEFAULT_WORKSPACE,
    extras: str = "all",
    *,
    install: bool = True,
) -> Environment:
    """Mount Drive, install the package, create the workspace, and report.

    Args:
        repo_root: Path to the cloned repository.
        workspace: Drive-backed directory for all outputs.
        extras: Optional-dependency group to install.
        install: Set False to skip installation on a warm restart where the
            package is already present, saving a few minutes.

    Returns:
        The environment snapshot, to be written into the run's provenance
        record.
    """
    repo_root = Path(repo_root)
    workspace = Path(workspace)

    mount_drive()
    if install:
        install_package(repo_root, extras=extras)
    create_workspace(workspace)

    environment = describe_environment(repo_root, workspace)

    print("\n" + "=" * 68)
    print("  GeoQ environment")
    print("=" * 68)
    for key, value in asdict(environment).items():
        if key == "packages":
            continue
        print(f"  {key:<16} {value}")
    print("-" * 68)
    for package, package_version in environment.packages.items():
        print(f"  {package:<16} {package_version}")
    print("=" * 68)

    if environment.git_dirty:
        print(
            "\n  WARNING: the working tree has uncommitted changes.\n"
            "  The recorded commit hash does not describe the code that is "
            "about to run,\n  so this result will not be reproducible. Commit "
            "before launching a long run."
        )

    return environment


if __name__ == "__main__":
    ENV = setup()
