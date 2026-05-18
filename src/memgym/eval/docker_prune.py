"""Quick utility to prune dangling Docker images."""
import subprocess
import sys

def main():
    result = subprocess.run(
        ["docker", "system", "prune", "-f"],
        capture_output=True, text=True, timeout=600,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
