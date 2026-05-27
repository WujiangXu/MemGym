"""
SWE-Gym test specs for the swebench evaluation harness.

The standard swebench package (v4.1.0) only has test specs for 66 repos.
SWE-Gym uses 11 additional repos that are missing from the standard mapping.
This module provides a monkey-patch function to add these specs at runtime,
making MemGym work as a general solution without requiring a forked swebench.

Source: https://github.com/SWE-Gym/SWE-Bench-Fork/blob/main/swebench/harness/constants.py
"""

import re

_patched = False


# =============================================================================
# SWE-Gym extra repo specs (from SWE-Gym/SWE-Bench-Fork)
# =============================================================================

# python/mypy
SPECS_MYPY = {
    k: {
        "pre_install": ["git submodule update --init mypy/typeshed || true"],
        "python": "3.12",
        "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r",
        "test_cmd": "pytest -rA -k",
    }
    for k in ["1.7", "1.8", "1.9", "1.10", "1.11"]
}
SPECS_MYPY.update({
    k: {
        "pre_install": ["git submodule update --init mypy/typeshed || true"],
        "python": "3.11",
        "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r",
        "test_cmd": "pytest -n0 -rA -k",
    }
    for k in ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
})
SPECS_MYPY.update({
    k: {
        "pre_install": ["git submodule update --init mypy/typeshed || true"],
        "python": "3.10",
        "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r",
        "test_cmd": "pytest -n0 -rA -k",
    }
    for k in ["0.990", "0.980", "0.970", "0.960", "0.950", "0.940"]
})
SPECS_MYPY.update({
    k: {
        "pre_install": [
            "git submodule update --init mypy/typeshed || true",
            "sed -i '1i types-typing-extensions==3.7.3' test-requirements.txt",
        ],
        "python": "3.9",
        "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r;",
        "test_cmd": "pytest -n0 -rA -k",
    }
    for k in ["0.920", "0.910", "0.820", "0.810", "0.800"]
})

# getmoto/moto
TEST_MOTO = "pytest -n0 -rA"
SPECS_MOTO = {
    k: {
        "python": "3.12",
        "install": "make init",
        "test_cmd": TEST_MOTO,
    }
    for k in [
        "0.4", "1.0", "1.2", "1.3",
        "2.0", "2.1", "2.2", "2.3",
        "3.0", "3.1",
        "4.0", "4.1", "4.2", "5.0",
    ]
}

# conan-io/conan
TEST_CONAN = "pytest -n0 -rA"
SPECS_CONAN = {
    k: {
        "python": "3.10",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential cmake",
        ],
        "install": "echo 'cython<3' > /tmp/constraint.txt; export PIP_CONSTRAINT=/tmp/constraint.txt; python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; python -m pip install -r conans/requirements_dev.txt ",
        "eval_commands": ["export PYTHONPATH=${PYTHONPATH:-}:$(pwd)"],
        "test_cmd": TEST_CONAN,
    }
    for k in ["1.33", "1.34", "1.36", "2.0", "1.35", "1.37", "1.46", "1.38", "1.39", "1.40", "1.41", "1.42", "1.45", "1.43", "1.44", "1.47", "1.48", "1.49", "1.50", "1.51", "1.52", "1.53", "1.55", "1.54", "1.57", "1.58", "1.59"]
}
SPECS_CONAN.update({
    k: {
        "python": "3.10",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential cmake",
        ],
        "install": "python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; python -m pip install -r conans/requirements_dev.txt ",
        "eval_commands": ["export PYTHONPATH=${PYTHONPATH:-}:$(pwd)"],
        "test_cmd": TEST_CONAN,
    }
    for k in ["2.1", "1.60", "1.61", "1.62", "2.2", "2.3", "2.4"]
})

# dask/dask
TEST_DASK = "pytest -n0 -rA  --color=no"
SPECS_DASK = {
    k: {
        "python": "3.10",
        "env_patches": ["sed -i '/- pip:/,/^ *-/d' environment.yml"],
        "packages": "environment.yml",
        "install": "python -m pip install --no-deps -e .",
        "test_cmd": TEST_DASK,
    }
    for k in ["2.11", "2.12", "2.13", "2.14", "2.15", "2.16", "2.17", "2.18", "2.19", "2.21", "2.22", "2.23", "2.25", "2.26", "2.27", "2.28", "2.29", "2.30", "2020.12", "2021.01", "2021.02", "2021.03", "2021.04", "2021.05", "2021.06", "2021.07", "2021.08", "2021.09", "2021.10", "2021.11", "2021.12", "2022.01", "2022.02", "2022.03", "2022.04", "2022.05", "2022.6", "2022.7", "2022.8", "2022.9", "2022.10", "2022.11", "2022.12", "2023.1", "2023.2", "2023.3", "2023.4", "2023.5", "2023.6", "2023.7", "2023.8", "2023.9", "2023.10", "2023.11", "2023.12", "2024.1", "2024.2", "2024.3", "2024.4", "2024.5"]
}

# Project-MONAI/MONAI
TEST_MONAI = "pytest -rA "
SPECS_MONAI = {
    k: {
        "python": "3.8",
        "install": "sed -i '/^git+https:\\/\\/github.com\\/Project-MONAI\\//d' requirements-dev.txt; python -m pip install types-pkg-resources==0.1.3 pytest; pip install -r requirements-dev.txt;python setup.py develop;",
        "test_cmd": TEST_MONAI,
    }
    for k in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.11", "0.105", "1.0", "1.1", "1.2", "1.3"]
}

# iterative/dvc
TEST_DVC = "pytest -rA"
SPECS_DVC = {
    k: {
        "python": "3.10",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y cmake",
            '[ -f setup.py ] && sed -E -i \'s/moto==([0-9]+\\.[0-9]+\\.[0-9]+)\\.dev[0-9]+/moto==\\1/\' setup.py',
            "[ -f setup.py ] && sed -i 's/pyarrow==0.15.1/pyarrow==0.16/' setup.py"
            "[ -f setup.py ] && sed -i 's/boto3==1.9.115/boto3==1.9.201/' setup.py",
        ],
        "install": 'python -m pip install --upgrade pip wheel GitPython; python -m pip install "cython<3.0.0" && python -m pip install --no-build-isolation pyyaml==5.4.1; python -m pip install git+https://github.com/iterative/mock-ssh-server.git || true; python -m pip install -r tests/requirements.txt || true; python -m pip install -r test-requirements.txt || true; python -m pip install -e ".[tests,dev,all_remotes,all,testing]";',
        "test_cmd": TEST_DVC,
    }
    for k in ["0.1", "0.8", "0.9", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20", "0.21", "0.22", "0.23", "0.24", "0.27", "0.28", "0.29", "0.30", "0.31", "0.32", "0.33", "0.34", "0.35", "0.40", "0.41", "0.50", "0.51", "0.52", "0.53", "0.54", "0.55", "0.56", "0.57", "0.58", "0.59", "0.60", "0.61", "0.62", "0.63", "0.65", "0.66", "0.68", "0.69", "0.70", "0.71", "0.74", "0.75", "0.76", "0.77", "0.78", "0.80", "0.81", "0.82", "0.83", "0.84", "0.85", "0.86", "0.87", "0.88", "0.89", "0.90", "0.91", "0.92", "0.93", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13", "2.15", "2.17", "2.19", "2.20", "2.21", "2.22", "2.23", "2.24", "2.27", "2.28", "2.30", "2.33", "2.34", "2.35", "2.38", "2.41", "2.43", "2.44", "2.45", "2.46", "2.48", "2.50", "2.51", "2.52", "2.54", "2.55", "2.56", "2.57", "2.58", "3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.10", "3.11", "3.12", "3.13", "3.14", "3.15", "3.17", "3.19", "3.23", "3.24", "3.28", "3.29", "3.36", "3.37", "3.38", "3.43", "3.47", "3.48", "3.49"]
}
for k in ["0.1", "0.8", "0.9", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20", "0.21", "0.22", "0.23", "0.24", "0.27", "0.28", "0.29", "0.30", "0.31", "0.32", "0.33", "0.34", "0.35", "0.40", "0.41", "0.50", "0.51", "0.52", "0.53", "0.54", "0.55", "0.56", "0.57", "0.58", "0.59", "0.60", "0.61", "0.62", "0.63", "0.65", "0.66", "0.68", "0.69", "0.70", "0.71", "0.74", "0.75", "0.76", "0.77", "0.78", "0.80", "0.81", "0.82", "0.83", "0.84", "0.85", "0.86", "0.87", "0.88", "0.89", "0.90", "0.91", "0.92", "0.93"]:
    SPECS_DVC[k]["python"] = "3.8"
    SPECS_DVC[k]["install"] += ' python -m pip install "numpy<=1.20";'
    SPECS_DVC[k]["install"] += ' python -m pip install "pytest<8";'
for k in ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13", "2.15", "2.17", "2.19", "2.20", "2.21", "2.22", "2.23", "2.24", "2.27", "2.28", "2.30", "2.33", "2.34", "2.35", "2.38", "2.41", "2.43", "2.44", "2.45", "2.46", "2.48", "2.50", "2.51", "2.52", "2.54", "2.55", "2.56", "2.57", "2.58", "3.0", "3.1", "3.2", "3.3"]:
    SPECS_DVC[k]["python"] = "3.9"
    SPECS_DVC[k]["install"] += ' python -m pip install "numpy<=1.20";'
    SPECS_DVC[k]["install"] += ' python -m pip install "pytest<8";'

# bokeh/bokeh
TEST_BOKEH = "pytest -rA -n0"
SPECS_BOKEH = {
    k: {
        "python": "3.10",
        "packages": "environment.yml",
        "pre_install": ["cd bokehjs && npm install --location=global npm && npm ci && cd ../"],
        "install": "python -m pip install -e .; python -m pip install bokeh_sampledata;",
        "test_cmd": TEST_BOKEH,
    }
    for k in ["3.0", "3.3", "3.4", "3.5"]
}
SPECS_BOKEH.update({
    k: {
        "python": "3.8",
        "packages": "environment.yml",
        "env_patches": [': "${CONDA_MKL_INTERFACE_LAYER_BACKUP:=\'\'}"'],
        "pre_install": ["cd bokehjs && npm install --location=global npm && npm ci && cd ../"],
        "install": 'pip install "setuptools<66" "jinja2<3.1"; printf "1\\n" | python setup.py develop; bokeh sampledata;',
        "test_cmd": TEST_BOKEH,
    }
    for k in ["2.0", "2.1", "2.3", "2.4"]
})
SPECS_BOKEH.update({
    k: {
        "python": "3.8",
        "packages": "environment.yml",
        "env_patches": [': "${CONDA_MKL_INTERFACE_LAYER_BACKUP:=\'\'}"'],
        "pre_install": ["cd bokehjs && npm install --location=global npm && npm ci && cd ../"],
        "install": 'pip install "setuptools<66" "jinja2<3.1"; printf "1\\n" | python setup.py develop; bokeh sampledata;',
        "test_cmd": TEST_BOKEH,
    }
    for k in ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.10", "0.11", "0.12", "0.13", "0.1181316818", "1.0", "1.1", "1.2", "1.3", "1.4"]
})

# modin-project/modin
TEST_MODIN = "pytest -n0 -rA"
SPECS_MODIN = {
    k: {
        "python": "3.9",
        "pre_install": ["apt-get -y update && apt-get -y upgrade && apt-get install -y libpq-dev"],
        "packages": "environment.yml",
        "install": "python -m pip install -e .;",
        "test_cmd": TEST_MODIN,
    }
    for k in ["0.1", "0.2", "0.3", "0.4", "0.6", "0.8", "0.9", "0.10", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20", "0.21", "0.22", "0.23", "0.24", "0.25", "0.26", "0.27", "0.28", "0.29", "0.30"]
}
for k in ["0.1", "0.2", "0.3", "0.4", "0.6", "0.8", "0.9", "0.10", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19"]:
    SPECS_MODIN[k]["python"] = "3.8"
    SPECS_MODIN[k]["install"] += " python -m pip install numpy==1.23.1 protobuf==3.20.1;"

# pydantic/pydantic
TEST_PYDANTIC = 'pytest -rA --tb=short -vv -o console_output_style=classic --no-header'
SPECS_PYDANTIC = {
    k: {
        "python": "3.8",
        "pre_install": [
            "apt-get update && apt-get install -y locales",
            "apt-get install -y pipx",
            "pipx ensurepath",
            "pipx install pdm",
            'export PATH="$HOME/.local/bin:$PATH"',
            "which python",
            "python --version",
        ],
        "install": 'export PATH="$HOME/.local/bin:$PATH"; pdm add pre-commit; make install;',
        "test_cmd": TEST_PYDANTIC,
    }
    for k in ["0.2", "0.41", "0.4", "0.6", "0.9", "0.10", "0.11", "0.13", "0.14", "0.151", "0.15", "0.17", "0.18", "0.201", "0.20", "0.24", "0.27", "0.29", "1.01", "0.32", "1.4", "1.31", "1.41", "1.51", "1.5", "1.71", "1.6", "1.7", "1.8", "1.9", "1.10", "2.0", "2.01", "2.02", "2.03", "2.04", "2.6", "2.5", "2.4", "2.7"]
}
for k in ["0.2", "0.41", "0.4", "0.6", "0.9", "0.10", "0.11", "0.13", "0.14", "0.151", "0.15", "0.17", "0.18", "0.201", "0.20", "0.24", "0.27", "0.29", "1.01", "0.32", "1.4", "1.31", "1.41", "1.51", "1.5", "1.71", "1.6", "1.7", "1.8", "1.9", "1.10"]:
    SPECS_PYDANTIC[k]["pre_install"] = [
        "apt-get update && apt-get install -y locales",
        "apt-get install -y pipx",
        "pipx ensurepath",
        "pipx install pdm  --python python3.7",
        'export PATH="$HOME/.local/bin:$PATH"',
        "which python",
        "python --version",
    ]
    SPECS_PYDANTIC[k]["python"] = "3.7"

# pandas-dev/pandas
TEST_PANDAS = "pytest -rA --tb=long"
SPECS_PANDAS = {
    k: {
        "python": "3.10",
        "packages": "environment.yml",
        "pre_install": [
            "git remote add upstream https://github.com/pandas-dev/pandas.git",
            "git fetch upstream --tags",
        ],
        "install": "python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true; pip uninstall pytest-qt -y;",
        "test_cmd": TEST_PANDAS,
    }
    for k in ["0.16", "0.17", "0.18", "0.19", "0.20", "0.21", "0.22", "0.23", "0.24", "0.25", "0.26", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "2.0", "2.1", "2.2", "3.0"]
}
for k in ["0.16", "0.17", "0.18", "0.19", "0.20", "0.21", "0.22", "0.23", "0.24", "0.25", "0.26", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "2.0", "2.1"]:
    SPECS_PANDAS[k]["install"] = "python -m pip install 'numpy<2'; " + SPECS_PANDAS[k]["install"]

# facebookresearch/hydra
TEST_HYDRA = "pytest -rA --tb=long"
SPECS_HYDRA = {
    k: {
        "python": "3.8",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y openjdk-17-jdk openjdk-17-jre",
        ],
        "install": "pip install -r requirements/dev.txt; pip install -e .;",
        "test_cmd": TEST_HYDRA,
    }
    for k in ["0.1", "0.9", "0.10", "0.11", "0.12", "1.0", "1.1", "1.2", "1.3", "1.4"]
}
for k in ["0.1", "0.9", "0.10", "0.11", "0.12", "1.0", "1.1", "1.2"]:
    SPECS_HYDRA[k]["install"] = (
        '{ tail -n1 requirements/requirements.txt | grep -q "." && echo ""; } >> requirements/requirements.txt; echo "pip==24.0" >> requirements/requirements.txt;'
        + 'pip install "pip==24.0"; '
        + "sed -i 's|isort@git+git://github.com/timothycrosley/isort|isort@git+https://github.com/timothycrosley/isort|g' requirements/dev.txt; "
        + SPECS_HYDRA[k]["install"]
    )


# =============================================================================
# Aggregated specs for all SWE-Gym repos
# =============================================================================

SWEGYM_REPO_SPECS = {
    "python/mypy": SPECS_MYPY,
    "getmoto/moto": SPECS_MOTO,
    "conan-io/conan": SPECS_CONAN,
    "dask/dask": SPECS_DASK,
    "Project-MONAI/MONAI": SPECS_MONAI,  # Mixed case matches HF dataset
    "iterative/dvc": SPECS_DVC,
    "bokeh/bokeh": SPECS_BOKEH,
    "modin-project/modin": SPECS_MODIN,
    "pydantic/pydantic": SPECS_PYDANTIC,
    "pandas-dev/pandas": SPECS_PANDAS,
    "facebookresearch/hydra": SPECS_HYDRA,
}

SWEGYM_REQS_PATHS = {
    "Project-MONAI/MONAI": ["requirements-dev.txt"],
    "facebookresearch/hydra": ["requirements/dev.txt"],
}

SWEGYM_ENV_YML_PATHS = {
    "bokeh/bokeh": ["conda/environment-test-3.10.yml", "environment.yml"],
    "modin-project/modin": ["environment-dev.yml"],
    "dask/dask": [
        "continuous_integration/environment-3.10.yaml",
        "continuous_integration/environment-3.9.yaml",
        "continuous_integration/environment-3.8.yaml",
        "continuous_integration/travis/travis-37.yaml",
    ],
    "pandas-dev/pandas": ["environment.yml"],
}

# All SWE-Gym repos are Python
SWEGYM_REPO_EXTS = {repo: "py" for repo in SWEGYM_REPO_SPECS}


# =============================================================================
# Mypy test command fix
# =============================================================================

def _make_mypy_test_command(instance):
    """Build a correct pytest -k command for mypy instances.

    Mypy test cases are defined as ``[case CaseName]`` blocks inside test files.
    The SWE-Bench-Fork extracts these case names and builds a keyword expression
    so pytest can filter by name, e.g. ``pytest -rA -k "CaseA or CaseB"``.

    The standard swebench harness would instead produce the broken command
    ``pytest -rA -k mypy/test/testcheck.py`` (file path after -k).
    """
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    test_cmd = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
    # Extract [case ...] names from the test patch
    case_names = re.findall(r'\[case ([^\]]+)\]', instance["test_patch"])
    if case_names:
        keyword_expr = " or ".join(case_names)
        return f'{test_cmd} "{keyword_expr}"'
    # Fallback: use file paths (strip the trailing -k to avoid broken syntax)
    from swebench.harness.test_spec.python import get_test_directives
    directives = get_test_directives(instance)
    base_cmd = test_cmd.rstrip().rsplit("-k", 1)[0].rstrip()
    return " ".join([base_cmd, *directives])


def _make_eval_script_mypy(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """Replacement eval script builder for mypy instances with correct -k handling."""
    from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
    from swebench.harness.utils import get_modified_files

    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = get_modified_files(test_patch)
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    test_command = _make_mypy_test_command(instance)

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]
    return eval_commands


def _make_eval_script_swegym(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """Eval script builder for SWE-Gym repos with conda set -u fix.

    The standard swebench eval script uses `set -uxo pipefail`. The -u (nounset)
    flag causes conda activation to crash on repos whose environments include
    binutils (pandas, etc.), because conda's activate-binutils_linux-64.sh
    references unset variables like $ADDR2LINE.

    Fix: wrap `conda activate` calls with `set +u` / `set -u`.
    """
    from swebench.harness.constants import (
        MAP_REPO_VERSION_TO_SPECS,
        START_TEST_OUTPUT,
        END_TEST_OUTPUT,
    )
    from swebench.harness.test_spec.python import get_test_directives
    from swebench.harness.utils import get_modified_files

    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = get_modified_files(test_patch)
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    test_cmd = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
    directives = get_test_directives(instance)
    test_command = " ".join([test_cmd, *directives])

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        "set +u",  # disable nounset — conda activation scripts may reference unset vars
        f"conda activate {env_name}",
        "set -u",  # re-enable nounset
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        "set +u",
        f"conda activate {env_name}",
        "set -u",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]
    return eval_commands


# =============================================================================
# Public API
# =============================================================================

def patch_swebench_specs():
    """
    Monkey-patch the swebench harness to add SWE-Gym repo test specs.

    Safe to call multiple times (idempotent). Patches:
    - MAP_REPO_VERSION_TO_SPECS (test/install commands per repo+version)
    - MAP_REPO_TO_REQS_PATHS (requirements file paths)
    - MAP_REPO_TO_ENV_YML_PATHS (conda environment file paths)
    - MAP_REPO_TO_EXT (repo language extension)
    - MAP_REPO_TO_PARSER (log parsers for grading)
    - TestSpec.instance_image_key (xingyaoww uses _s_ not _1776_ for __ in image tags)
    """
    global _patched
    if _patched:
        return

    from swebench.harness.constants import (
        MAP_REPO_VERSION_TO_SPECS,
        MAP_REPO_TO_REQS_PATHS,
        MAP_REPO_TO_ENV_YML_PATHS,
        MAP_REPO_TO_EXT,
    )

    added = []
    for repo, specs in SWEGYM_REPO_SPECS.items():
        if repo not in MAP_REPO_VERSION_TO_SPECS:
            MAP_REPO_VERSION_TO_SPECS[repo] = specs
            added.append(repo)

    for repo, ext in SWEGYM_REPO_EXTS.items():
        if repo not in MAP_REPO_TO_EXT:
            MAP_REPO_TO_EXT[repo] = ext

    for repo, paths in SWEGYM_REQS_PATHS.items():
        if repo not in MAP_REPO_TO_REQS_PATHS:
            MAP_REPO_TO_REQS_PATHS[repo] = paths

    for repo, paths in SWEGYM_ENV_YML_PATHS.items():
        if repo not in MAP_REPO_TO_ENV_YML_PATHS:
            MAP_REPO_TO_ENV_YML_PATHS[repo] = paths

    # Patch log parsers for grading — all SWE-Gym repos use pytest.
    # Use parse_log_pytest_v2 which handles BOTH output formats:
    #   "PASSED tests/test_foo.py::test_bar"  (status first — older pytest)
    #   "tests/test_foo.py::test_bar PASSED"  (path first — newer pytest, e.g. pydantic)
    # The original parse_log_pytest only matches lines starting with PASSED/FAILED,
    # which silently returns {} for repos like pydantic that use the path-first format.
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.log_parsers.python import parse_log_pytest_v2
    for repo in SWEGYM_REPO_SPECS:
        if repo not in MAP_REPO_TO_PARSER:
            MAP_REPO_TO_PARSER[repo] = parse_log_pytest_v2

    # Monkey-patch instance_image_key to handle xingyaoww's _s_ naming convention.
    # Standard swebench v4.1.0 replaces __ with _1776_ for remote images, but
    # xingyaoww (SWE-Gym) pushed images with _s_ replacement instead.
    from swebench.harness.test_spec.test_spec import TestSpec

    def _patched_instance_image_key(self):
        key = f"sweb.eval.{self.arch}.{self.instance_id.lower()}:{self.instance_image_tag}"
        if self.is_remote_image:
            key = f"{self.namespace}/{key}"
            if self.namespace and "xingyaoww" in self.namespace:
                key = key.replace("__", "_s_")
            else:
                key = key.replace("__", "_1776_")
        return key

    TestSpec.instance_image_key = property(_patched_instance_image_key)

    # Monkey-patch the eval script builder to fix mypy test command construction.
    # The standard swebench harness concatenates test_cmd + file paths from
    # get_test_directives(). For mypy, test_cmd ends with "-k" which expects
    # keyword expressions, not file paths. The SWE-Bench-Fork handles this by
    # parsing [case CaseName] patterns from the test_patch and building a proper
    # keyword expression like: pytest -rA -k "CaseA or CaseB"
    import swebench.harness.test_spec.python as _py_mod
    import swebench.harness.test_spec.create_scripts as _cs_mod
    _original_make_eval_script_list_py = _py_mod.make_eval_script_list_py

    def _patched_make_eval_script_list_py(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    ):
        if instance["repo"] == "python/mypy":
            return _make_eval_script_mypy(
                instance, specs, env_name, repo_directory, base_commit, test_patch
            )
        if instance["repo"] in SWEGYM_REPO_SPECS:
            return _make_eval_script_swegym(
                instance, specs, env_name, repo_directory, base_commit, test_patch
            )
        return _original_make_eval_script_list_py(
            instance, specs, env_name, repo_directory, base_commit, test_patch
        )

    # Patch both the source module and the already-imported reference in create_scripts
    _py_mod.make_eval_script_list_py = _patched_make_eval_script_list_py
    _cs_mod.make_eval_script_list_py = _patched_make_eval_script_list_py

    # Monkey-patch docker.from_env() to use a longer HTTP timeout.
    # The Docker Python SDK defaults to DEFAULT_TIMEOUT_SECONDS=60 (in docker/constants.py).
    # When swebench calls client.images.pull() for remote images (xingyaoww namespace),
    # pulling 500MB-2GB images over the network often exceeds 60s, causing:
    #   "UnixHTTPConnectionPool(host='localhost'): Read timed out. (read timeout=60)"
    # This affects ALL pandas, pydantic, and ~44% of mypy instances in our runs.
    # Fix: wrap docker.from_env() to pass timeout=600 (10 minutes).
    import docker as _docker_mod
    _original_docker_from_env = _docker_mod.from_env

    def _docker_from_env_with_timeout(**kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600  # 10 minutes instead of 60 seconds
        return _original_docker_from_env(**kwargs)

    _docker_mod.from_env = _docker_from_env_with_timeout

    _patched = True
    if added:
        print(f"[MemGym] Patched swebench with {len(added)} SWE-Gym repo specs: {', '.join(added)}")
    print("[MemGym] Patched eval scripts: mypy -k fix + conda set +u for all SWE-Gym repos")
    print("[MemGym] Patched log parser: parse_log_pytest_v2 (handles path-first format)")
    print("[MemGym] Patched docker.from_env() timeout: 60s → 600s")
