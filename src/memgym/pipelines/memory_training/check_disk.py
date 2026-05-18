import subprocess, os, json, sys

# If --git-update flag, do git fetch+reset first
if "--git-update" in sys.argv:
    print("=== GIT UPDATE ===")
    r1 = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
    print(f"Before: {r1.stdout.strip()}")
    subprocess.run(["git", "fetch", "origin"], capture_output=True, text=True)
    r2 = subprocess.run(["git", "reset", "--hard", "origin/master"], capture_output=True, text=True)
    print(f"Reset: {r2.stdout.strip()}")
    r3 = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
    print(f"After: {r3.stdout.strip()}")

# If --pull-images flag, pull missing docker images from preds.json
if "--pull-images" in sys.argv:
    preds_idx = sys.argv.index("--pull-images") + 1
    preds_path = sys.argv[preds_idx] if preds_idx < len(sys.argv) else "results/trajectories_all/baseline/preds.json"
    workers = 4
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    print(f"\n=== PULLING MISSING IMAGES from {preds_path} ===")
    preds = json.load(open(preds_path))
    error_ids = [iid for iid, p in preds.items() if p.get("status") == "Error"]

    # Get image names
    images = []
    for iid in sorted(error_ids):
        docker_id = iid.replace("__", "_s_")
        images.append(f"xingyaoww/sweb.eval.x86_64.{docker_id}:latest")

    # Check local
    r = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True)
    local = set(r.stdout.strip().split("\n")) if r.stdout.strip() else set()
    to_pull = [img for img in images if img not in local]
    print(f"Error instances: {len(error_ids)}, already local: {len(images)-len(to_pull)}, to pull: {len(to_pull)}")

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def pull_one(args):
        idx, img = args
        try:
            r = subprocess.run(["docker", "pull", img], capture_output=True, text=True, timeout=600)
            ok = r.returncode == 0
            if ok:
                print(f"[{idx}/{len(to_pull)}] OK: {img}")
            else:
                print(f"[{idx}/{len(to_pull)}] FAIL: {img} — {r.stderr[:100]}")
            return (img, ok)
        except Exception as e:
            print(f"[{idx}/{len(to_pull)}] ERROR: {img} — {e}")
            return (img, False)

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(pull_one, (i+1, img)): img for i, img in enumerate(to_pull)}
        for f in as_completed(futs):
            results.append(f.result())
    elapsed = time.time() - t0
    ok = sum(1 for _, s in results if s)
    fail = sum(1 for _, s in results if not s)
    print(f"Done in {elapsed:.0f}s: {ok} pulled, {fail} failed")
    sys.exit(0)

# Disk usage
df = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
print("=== DISK USAGE ===")
print(df.stdout)

# Docker system df
docker_df = subprocess.run(["docker", "system", "df"], capture_output=True, text=True)
print("=== DOCKER STORAGE ===")
print(docker_df.stdout)

# Count Docker images
docker_imgs = subprocess.run(["docker", "images", "--format", "{{.Repository}}\t{{.Size}}"], capture_output=True, text=True)
lines = docker_imgs.stdout.strip().split("\n") if docker_imgs.stdout.strip() else []
print(f"=== DOCKER IMAGES: {len(lines)} total ===")

# Group by prefix
prefixes = {}
total_count = 0
for line in lines:
    parts = line.split("\t")
    repo = parts[0] if parts else "unknown"
    size = parts[1] if len(parts) > 1 else "?"
    prefix = repo.split("/")[0] if "/" in repo else repo
    if prefix not in prefixes:
        prefixes[prefix] = {"count": 0, "sample_size": size}
    prefixes[prefix]["count"] += 1
    total_count += 1

for prefix, info in sorted(prefixes.items(), key=lambda x: -x[1]["count"]):
    print(f"  {prefix}: {info['count']} images (sample: {info['sample_size']})")

# Results directory size
du = subprocess.run(["du", "-sh", "results/"], capture_output=True, text=True)
print(f"\n=== RESULTS DIR SIZE ===")
print(du.stdout)

# Trajectories_all size
du2 = subprocess.run(["du", "-sh", "results/trajectories_all/"], capture_output=True, text=True)
print(f"=== TRAJECTORIES_ALL SIZE ===")
print(du2.stdout)
