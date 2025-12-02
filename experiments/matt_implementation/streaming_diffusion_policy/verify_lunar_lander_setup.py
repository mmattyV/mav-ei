"""
Verification script to check that all Lunar Lander files are properly set up.
Run this before attempting to collect data or train.

Usage:
    python verify_lunar_lander_setup.py
"""

import sys
import pathlib
from typing import List, Tuple

def check_file_exists(filepath: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    path = pathlib.Path(filepath)
    if path.exists():
        return True, f"✓ Found: {filepath}"
    else:
        return False, f"✗ Missing: {filepath}"

def check_import(module_name: str) -> Tuple[bool, str]:
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True, f"✓ Can import: {module_name}"
    except ImportError as e:
        return False, f"✗ Cannot import: {module_name} ({e})"

def main():
    print("=" * 60)
    print("Lunar Lander Setup Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check required files
    print("Checking required files...")
    print("-" * 60)
    
    required_files = [
        "scripts/collect_lunar_lander_demos.py",
        "diffusion_policy/dataset/lunar_lander_dataset.py",
        "diffusion_policy/env_runner/lunar_lander_runner.py",
        "diffusion_policy/config/task/lunar_lander_lowdim.yaml",
        "diffusion_policy/config/train_tedi_unet_lunar_lander.yaml",
        "setup_lunar_lander.sh",
        "LUNAR_LANDER_README.md",
    ]
    
    for filepath in required_files:
        passed, message = check_file_exists(filepath)
        print(message)
        all_checks_passed = all_checks_passed and passed
    
    print()
    
    # Check required Python packages
    print("Checking required packages...")
    print("-" * 60)
    
    required_packages = [
        "gymnasium",
        "zarr",
        "numpy",
        "torch",
        "tqdm",
        "hydra",
        "omegaconf",
        "wandb",
    ]
    
    for package in required_packages:
        passed, message = check_import(package)
        print(message)
        all_checks_passed = all_checks_passed and passed
    
    print()
    
    # Check optional packages (for data collection)
    print("Checking optional packages (for expert data collection)...")
    print("-" * 60)
    
    optional_packages = [
        "stable_baselines3",
        "huggingface_sb3",
    ]
    
    optional_all_good = True
    for package in optional_packages:
        passed, message = check_import(package)
        print(message)
        optional_all_good = optional_all_good and passed
    
    if not optional_all_good:
        print()
        print("⚠ Note: Optional packages missing. You can still collect data")
        print("  using random policy, but expert demonstrations require:")
        print("  pip install stable-baselines3 huggingface-sb3")
    
    print()
    print("=" * 60)
    
    if all_checks_passed:
        print("✓ All required checks passed!")
        print()
        print("Next steps:")
        print("  1. Collect data:")
        print("     ./setup_lunar_lander.sh")
        print("     OR")
        print("     python scripts/collect_lunar_lander_demos.py")
        print()
        print("  2. Train SDP:")
        print("     python train.py --config-dir=diffusion_policy/config/")
        print("     --config-name=train_tedi_unet_lunar_lander")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Install packages: pip install gymnasium[box2d] zarr tqdm")
        print("  - Check that you're in the streaming_diffusion_policy directory")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())



