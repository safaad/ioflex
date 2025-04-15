#!/usr/bin/env python3

import os
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
SKIP_SMALL_FILE_SIZE = 1 * 1024 * 1024  # 1MB
DD_BLOCK_SIZE = "4M"
MAX_WORKERS = 8

# === LOGGING ===
logging.basicConfig(
    filename="restriping.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def run_lfs_setstripe(path: str, stripe_count: int, stripe_size: str = None):
    cmd = ["lfs", "setstripe", "-c", str(stripe_count)]
    if stripe_size:
        cmd += ["-S", stripe_size]
    cmd.append(path)
    subprocess.run(cmd, check=True)


def copy_file(src: Path, dest: Path):
    try:

        subprocess.run(
            [
                "dd",
                f"if={src}",
                f"of={dest}",
                f"bs={DD_BLOCK_SIZE}",
                "oflag=direct",
                "status=none",
            ],
            check=True,
        )
        shutil.copy2(src, dest)
    except subprocess.CalledProcessError:
        logging.warning(f"dd failed fos {src}, falling back to shutil.copy2")
        shutil.copy2(src, dest)


def restripe_file(
    src_file: Path, tmp_base: Path, stripe_count: int, stripe_size: str = None
):
    rel_path = src_file.relative_to(src_file.anchor)
    tmp_file = tmp_base / rel_path

    try:
        if src_file.stat().st_size < SKIP_SMALL_FILE_SIZE:
            logging.info(f"Skipping small file: {src_file}")
            return

        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        run_lfs_setstripe(str(tmp_file), stripe_count, stripe_size)

        copy_file(src_file, tmp_file)

        tmp_file.replace(src_file)
        logging.info(f"Re-striped: {src_file}")
    except Exception as e:
        logging.error(f"Failed to restripe {src_file}: {e}")


def restripe_directory(target_dir: Path, stripe_count: int, stripe_size: str = None):
    tmp_dir = target_dir.parent / (target_dir.name + "_tmp_stripe")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = []

    for root, dirs, files in os.walk(target_dir):
        # Set stripe on subdirs too
        for d in dirs:
            dir_path = Path(root) / d
            try:
                run_lfs_setstripe(str(dir_path), stripe_count, stripe_size)
                logging.info(f"Set stripe on dir: {dir_path}")
            except subprocess.CalledProcessError:
                logging.warning(f"Failed to set stripe on dir: {dir_path}")

        for f in files:
            src_file = Path(root) / f
            files_to_process.append((src_file, tmp_dir, stripe_count, stripe_size))

    # Parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(restripe_file, *args) for args in files_to_process]
        for future in as_completed(futures):
            future.result()

    shutil.rmtree(tmp_dir)
    logging.info(f"Completed re-striping for directory: {target_dir}")


def setstriping(path, stripe_count, stripe_size):
    target = Path(path).resolve()

    if not target.exists():
        print(f"Error: Path does not exist: {target}")
        return
    try:
        if target.is_file():
            tmp_dir = target.parent / ".tmp_stripe"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            restripe_file(target, tmp_dir, stripe_count, stripe_size)
            shutil.rmtree(tmp_dir)
        elif target.is_dir():
            restripe_directory(target, stripe_count, stripe_size)
        else:
            print("Unsupported path type.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Lustre striping for existing files/directories."
    )
    parser.add_argument("path", help="Target file or directory")
    parser.add_argument(
        "-c", "--stripe-count", type=int, required=True, help="Stripe count"
    )
    parser.add_argument("-s", "--stripe-size", help="Stripe size (e.g., 1M, 4M, etc.)")

    args = parser.parse_args()
    target = Path(args.path).resolve()

    if not target.exists():
        print(f"Error: Path does not exist: {target}")
        return

    try:
        if target.is_file():
            tmp_dir = target.parent / ".tmp_stripe"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            restripe_file(target, tmp_dir, args.stripe_count, args.stripe_size)
            shutil.rmtree(tmp_dir)
        elif target.is_dir():
            restripe_directory(target, args.stripe_count, args.stripe_size)
        else:
            print("Unsupported path type.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
