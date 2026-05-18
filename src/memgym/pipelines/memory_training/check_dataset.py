"""Check SWE-Gym dataset composition for a given slice."""
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--slice", default="0:500")
args = parser.parse_args()

start, end = map(int, args.slice.split(":"))

ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
print(f"Total SWE-Gym train instances: {len(ds)}")

repos = {}
for i in range(start, min(end, len(ds))):
    repo = ds[i]["repo"]
    repos[repo] = repos.get(repo, 0) + 1

print(f"\nSlice [{start}:{end}]: {sum(repos.values())} instances, {len(repos)} repos")
for repo, count in sorted(repos.items(), key=lambda x: -x[1]):
    print(f"  {repo}: {count}")

# Estimate Docker images needed
print(f"\nDocker images needed: ~{len(repos)} unique repo images")
print(f"Estimated storage: ~{len(repos) * 2.85:.0f} GB")
