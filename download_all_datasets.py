#!/usr/bin/env python3
"""
Batch download all required datasets

This script downloads all datasets defined for D4RL and OGBench.

Supported dataset types:
1. D4RL datasets: antmaze, pen, door, hammer, relocate series
2. OGBench datasets: various maze, cube, puzzle environments
"""

import os
import sys
import time
from typing import List, Dict

# All D4RL environment names
D4RL_DATASETS = [
    # From 11-antmaze-umaze_medium_large_v2.sh
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",

    # From 12-hammer-relocate-v1.sh
    "hammer-human-v1",
    "hammer-cloned-v1",
    "hammer-expert-v1",
    "relocate-human-v1",
    "relocate-cloned-v1",
    "relocate-expert-v1",

    # From 13-pen_door_v1.sh
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
    "door-human-v1",
    "door-cloned-v1",
    "door-expert-v1"
]

# All OGBench environment names
OGBENCH_NAMES = {
    'antmaze-giant-navigate-v0': None,
    'antmaze-giant-stitch-v0': None,
    'antmaze-large-explore-v0': None,
    'antmaze-large-navigate-v0': None,
    'antmaze-large-stitch-v0': None,
    'antmaze-medium-explore-v0': None,
    'antmaze-medium-navigate-v0': None,
    'antmaze-medium-stitch-v0': None,
    'antmaze-teleport-explore-v0': None,
    'antmaze-teleport-navigate-v0': None,
    'antmaze-teleport-stitch-v0': None,
    'antsoccer-arena-navigate-v0': None,
    'antsoccer-arena-stitch-v0': None,
    'antsoccer-medium-navigate-v0': None,
    'antsoccer-medium-stitch-v0': None,
    'cube-double-noisy-v0': None,
    'cube-double-play-v0': None,
    'cube-quadruple-noisy-v0': None,
    'cube-quadruple-play-v0': None,
    'cube-single-noisy-v0': None,
    'cube-single-play-v0': None,
    'cube-triple-noisy-v0': None,
    'cube-triple-play-v0': None,
    'humanoidmaze-giant-navigate-v0': None,
    'humanoidmaze-giant-stitch-v0': None,
    'humanoidmaze-large-navigate-v0': None,
    'humanoidmaze-large-stitch-v0': None,
    'humanoidmaze-medium-navigate-v0': None,
    'humanoidmaze-medium-stitch-v0': None,
    'pointmaze-giant-navigate-v0': None,
    'pointmaze-giant-stitch-v0': None,
    'pointmaze-large-navigate-v0': None,
    'pointmaze-large-stitch-v0': None,
    'pointmaze-medium-navigate-v0': None,
    'pointmaze-medium-stitch-v0': None,
    'pointmaze-teleport-navigate-v0': None,
    'pointmaze-teleport-stitch-v0': None,
    'powderworld-easy-play-v0': None,
    'powderworld-hard-play-v0': None,
    'powderworld-medium-play-v0': None,
    'puzzle-3x3-noisy-v0': None,
    'puzzle-3x3-play-v0': None,
    'puzzle-4x4-noisy-v0': None,
    'puzzle-4x4-play-v0': None,
    'puzzle-4x5-noisy-v0': None,
    'puzzle-4x5-play-v0': None,
    'puzzle-4x6-noisy-v0': None,
    'puzzle-4x6-play-v0': None,
    'scene-noisy-v0': None,
    'scene-play-v0': None,
    'visual-antmaze-giant-navigate-v0': None,
    'visual-antmaze-giant-stitch-v0': None,
    'visual-antmaze-large-explore-v0': None,
    'visual-antmaze-large-navigate-v0': None,
    'visual-antmaze-large-stitch-v0': None,
    'visual-antmaze-medium-explore-v0': None,
    'visual-antmaze-medium-navigate-v0': None,
    'visual-antmaze-medium-stitch-v0': None,
    'visual-antmaze-teleport-explore-v0': None,
    'visual-antmaze-teleport-navigate-v0': None,
    'visual-antmaze-teleport-stitch-v0': None,
    'visual-cube-double-noisy-v0': None,
    'visual-cube-double-play-v0': None,
    'visual-cube-quadruple-noisy-v0': None,
    'visual-cube-quadruple-play-v0': None,
    'visual-cube-single-noisy-v0': None,
    'visual-cube-single-play-v0': None,
    'visual-cube-triple-noisy-v0': None,
    'visual-cube-triple-play-v0': None,
    'visual-humanoidmaze-giant-navigate-v0': None,
    'visual-humanoidmaze-giant-stitch-v0': None,
    'visual-humanoidmaze-large-navigate-v0': None,
    'visual-humanoidmaze-large-stitch-v0': None,
    'visual-humanoidmaze-medium-navigate-v0': None,
    'visual-humanoidmaze-medium-stitch-v0': None,
    'visual-puzzle-3x3-noisy-v0': None,
    'visual-puzzle-3x3-play-v0': None,
    'visual-puzzle-4x4-noisy-v0': None,
    'visual-puzzle-4x4-play-v0': None,
    'visual-puzzle-4x5-noisy-v0': None,
    'visual-puzzle-4x5-play-v0': None,
    'visual-puzzle-4x6-noisy-v0': None,
    'visual-puzzle-4x6-play-v0': None,
    'visual-scene-noisy-v0': None,
    'visual-scene-play-v0': None
}

ALL_DATASETS = D4RL_DATASETS + list(OGBENCH_NAMES.keys())

def check_dependencies():
    """Check if necessary dependencies are installed"""
    print("🔍 Checking dependencies...")
    all_ok = True
    try:
        import gym
        print("✅ gym installed")
    except ImportError:
        print("❌ gym not installed, please run: pip install gym")
        all_ok = False

    try:
        import d4rl
        print("✅ d4rl installed")
    except ImportError:
        print("❌ d4rl not installed, please run: pip install git+https://github.com/Farama-Foundation/d4rl.git")
        all_ok = False

    try:
        import ogb
        print("✅ ogb installed")
    except ImportError:
        print("❌ ogb not installed, please run: pip install ogb")
        all_ok = False

    try:
        import mujoco_py
        print("✅ mujoco_py installed")
    except ImportError:
        print("⚠️  mujoco_py not installed, some environments may not work")

    return all_ok

def download_d4rl_dataset(env_name: str) -> bool:
    """
    Download D4RL dataset by creating the environment.
    D4RL datasets are automatically downloaded on first use.
    """
    print(f"📥 Preparing to download D4RL dataset: {env_name}")
    try:
        import gym
        import d4rl

        if any(keyword in env_name for keyword in ['pen', 'hammer', 'relocate', 'door']):
            import d4rl.hand_manipulation_suite  # noqa
            print(f"   Imported hand_manipulation_suite for {env_name}")

        env = gym.make(env_name)
        dataset = env.get_dataset()

        print(f"✅ {env_name} dataset ready")
        print(f"   - Observation shape: {dataset['observations'].shape}")
        print(f"   - Action shape: {dataset['actions'].shape}")
        print(f"   - Number of data points: {len(dataset['observations'])}")

        env.close()
        return True

    except Exception as e:
        print(f"❌ {env_name} download failed: {str(e)}")
        return False

def download_ogbench_datasets(dataset_names: List[str]) -> bool:
    """
    Download a list of OGBench datasets.
    """
    if not dataset_names:
        return True

    print(f"📥 Downloading {len(dataset_names)} OGBench datasets...")
    try:
        import ogbench
        dataset_dir = './dataset'
        os.makedirs(dataset_dir, exist_ok=True)

        ogbench.download_datasets(
            dataset_names,
            dataset_dir=dataset_dir,
        )

        print(f"✅ All OGBench datasets downloaded successfully to '{dataset_dir}'")
        return True

    except ImportError:
        print("⚠️  ogbench is not installed, skipping OGBench datasets. Please run `pip install ogbench`")
        return False
    except Exception as e:
        print(f"❌ OGBench download failed: {str(e)}")
        return False

def categorize_datasets(datasets: List[str]) -> Dict[str, List[str]]:
    """Categorize datasets by type"""
    d4rl_datasets = []
    ogbench_datasets = []

    for dataset in datasets:
        if dataset in D4RL_DATASETS:
            d4rl_datasets.append(dataset)
        elif dataset in OGBENCH_NAMES:
            ogbench_datasets.append(dataset)

    return {
        'd4rl': d4rl_datasets,
        'ogbench': ogbench_datasets
    }

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 Batch Dataset Download Tool")
    print("=" * 60)

    if not check_dependencies():
        print("\n💥 Dependency check failed, please install necessary packages first")
        sys.exit(1)

    categorized = categorize_datasets(ALL_DATASETS)

    print(f"\n📊 Dataset statistics:")
    print(f"   - D4RL datasets: {len(categorized['d4rl'])} items")
    print(f"   - OGBench datasets: {len(categorized['ogbench'])} items")
    print(f"   - Total: {len(ALL_DATASETS)} items")

    # --- Download D4RL datasets ---
    print(f"\n🔄 Starting D4RL dataset download...")
    d4rl_success_count = 0
    d4rl_failed_list = []

    for i, dataset in enumerate(categorized['d4rl'], 1):
        print(f"\n[{i}/{len(categorized['d4rl'])}] Processing D4RL: {dataset}")
        if download_d4rl_dataset(dataset):
            d4rl_success_count += 1
        else:
            d4rl_failed_list.append(dataset)
        time.sleep(1)

    # --- Download OGBench datasets ---
    ogbench_success = False
    if categorized['ogbench']:
        print(f"\n🔄 Starting OGBench dataset download...")
        if download_ogbench_datasets(categorized['ogbench']):
            ogbench_success = True

    # --- Output summary ---
    print("\n" + "=" * 60)
    print("📈 Download Summary")
    print("=" * 60)

    print(f"✅ D4RL datasets successful: {d4rl_success_count}/{len(categorized['d4rl'])}")
    if d4rl_failed_list:
        print(f"❌ D4RL datasets failed: {d4rl_failed_list}")

    if categorized['ogbench']:
        ogbench_status = "✅ Succeeded" if ogbench_success else "❌ Failed"
        print(f"{ogbench_status} for {len(categorized['ogbench'])} OGBench datasets.")

    total_d4rl = len(categorized['d4rl'])
    total_ogbench = len(categorized['ogbench'])
    total_success = d4rl_success_count + (total_ogbench if ogbench_success else 0)
    total_datasets = total_d4rl + total_ogbench

    if total_datasets > 0:
        print(f"\n🎯 Overall success rate: {total_success}/{total_datasets} ({total_success/total_datasets*100:.1f}%)")

    if total_success == total_datasets:
        print("\n🎉 All datasets are ready! You can now run training scripts.")
    else:
        print(f"\n⚠️  {total_datasets - total_success} datasets failed to download. Please check network connection and dependencies.")

if __name__ == "__main__":
    main()