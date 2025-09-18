import sys, os, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]  # go up from tests/ to repo root
sys.path.insert(0, str(repo_root))
