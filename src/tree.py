#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import fnmatch
from typing import List, Optional

# --- Constants for drawing the tree structure ---
PREFIX_BRANCH = "├── "
PREFIX_LAST = "└── "
PREFIX_PARENT = "│   "
PREFIX_EMPTY = "    "

# --- Default ignore patterns ---
DEFAULT_IGNORE_PATTERNS = [".venv*", "__pycache__", "*.pyc"]

def build_tree(
    target_dir: str,
    level: Optional[int],
    ignore_patterns: List[str],
    prefix: str = "",
):
    """
    Recursively scans a directory structure to generate and print a tree.
    """
    # Using scandir() is more efficient as it retrieves file type info during the scan.
    try:
        entries = sorted(
            [e for e in os.scandir(target_dir)], key=lambda e: e.name.lower()
        )
    except OSError as e:
        print(f"Error reading directory {target_dir}: {e}")
        return

    # Filter out entries that match any of the ignore patterns.
    filtered_entries = []
    for entry in entries:
        if not any(fnmatch.fnmatch(entry.name, pattern) for pattern in ignore_patterns):
            filtered_entries.append(entry)

    entry_count = len(filtered_entries)
    for i, entry in enumerate(filtered_entries):
        is_last = i == (entry_count - 1)
        connector = PREFIX_LAST if is_last else PREFIX_BRANCH
        print(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            # Check the depth limit (-L option).
            if level is None or level > 1:
                new_prefix = prefix + (PREFIX_EMPTY if is_last else PREFIX_PARENT)
                next_level = level - 1 if level is not None else None
                build_tree(entry.path, next_level, ignore_patterns, new_prefix)

def main():
    """
    Main function to parse command-line arguments and initiate the tree building.
    """
    parser = argparse.ArgumentParser(
        description="A Python script to display directory structure like the 'tree' command."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="The directory to start from (default: current directory).",
    )
    parser.add_argument(
        "-L",
        "--level",
        type=int,
        help="Descend only 'level' directories deep.",
    )
    parser.add_argument(
        "-I",
        "--ignore",
        type=str,
        help="Add wildcard patterns to the ignore list, separated by '|'. "
             "This is in addition to the default ignore patterns.",
    )

    args = parser.parse_args()

    # Start with a copy of the default ignore patterns.
    ignore_list = DEFAULT_IGNORE_PATTERNS[:]

    # If the user provided additional patterns with -I, add them to the list.
    if args.ignore:
        user_patterns = args.ignore.split("|")
        ignore_list.extend(user_patterns)

    start_dir = args.directory
    if not os.path.isdir(start_dir):
        print(f"Error: Directory '{start_dir}' not found.")
        return

    print(start_dir)
    build_tree(start_dir, args.level, ignore_list)

if __name__ == "__main__":
    main()