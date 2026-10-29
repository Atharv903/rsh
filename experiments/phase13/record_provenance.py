"""
Phase 13A: Provenance and Hash Integrity Verification Script.
Records SHA-256 checksums of all model checkpoints, configuration files,
scenario manifests, and evaluation scripts.
"""

import hashlib
import json
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def get_git_info():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    return {"commit": commit, "branch": branch, "dirty": len(status) > 0, "status": status}

def main():
    provenance = {
        "timestamp": "2026-10-29",
        "git": get_git_info(),
        "checkpoints": {},
        "configurations_and_manifests": {},
        "evaluation_scripts": {},
        "phase12_artifacts": {},
    }

    # Checkpoints
    ckpt_dir = repo_root / "experiments/phase8_corrected_speed_models/checkpoints"
    for pt_file in sorted(ckpt_dir.glob("*/*.pt")):
        rel_path = pt_file.relative_to(repo_root).as_posix()
        provenance["checkpoints"][rel_path] = {
            "size_bytes": pt_file.stat().st_size,
            "sha256": compute_sha256(pt_file),
        }

    # Manifests and configs
    manifest_files = [
        "experiments/phase12_validation/phase12_scenarios.json",
        "experiments/phase12_validation/experiment_config.json",
        "experiments/phase12_validation/candidate_sessions_audit.json",
        "experiments/phase12_validation/data_quality_audit.json",
        "experiments/phase10_independent_validation/phase10_scenarios.json",
    ]
    for mf in manifest_files:
        p = repo_root / mf
        if p.exists():
            provenance["configurations_and_manifests"][mf] = {
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }

    # Scripts
    script_files = [
        "src/outage_estimator.py",
        "scripts/evaluate_phase12_expanded.py",
        "scripts/audit_candidate_sessions_phase12.py",
        "scripts/plot_phase12_figures.py",
        "tests/test_navigation_heading.py",
        "scripts/evaluate_phase10_independent.py",
    ]
    for sf in script_files:
        p = repo_root / sf
        if p.exists():
            provenance["evaluation_scripts"][sf] = {
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }

    # Phase 12 results artifacts
    phase12_artifacts = [
        "experiments/phase12_validation/per_scenario_metrics.csv",
        "experiments/phase12_validation/per_session_summary.csv",
        "experiments/phase12_validation/statistical_comparison.csv",
        "reports/phase12_navigation_correction_and_validation.md",
    ]
    for af in phase12_artifacts:
        p = repo_root / af
        if p.exists():
            provenance["phase12_artifacts"][af] = {
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }

    out_file = repo_root / "experiments/phase13/provenance_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"Provenance saved to {out_file}")
    print(f"Git commit: {provenance['git']['commit']}")
    print(f"Total checkpoints audited: {len(provenance['checkpoints'])}")
    print(f"Total scripts audited: {len(provenance['evaluation_scripts'])}")

if __name__ == "__main__":
    main()
