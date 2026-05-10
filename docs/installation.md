# Installation

`ad` can be installed from PyPI or directly from source via GitHub.

---

## [PyPI](https://pypi.org/project/ad/)

For using the PyPI package in your project, add it to your configuration file:

=== "pyproject.toml"

    ```toml
    [project]
    dependencies = [
      "ad>=1.3.2", # (1)!
    ]
    ```

    1. Specifying a version is recommended.

=== "requirements.txt"

    ```text
    ad>=1.3.2
    ```

### pip

=== "Installation for user"

    ```bash
    pip install --upgrade --user ad # (1)!
    ```

    1. You may need to use `pip3` instead of `pip` depending on your Python installation.

=== "Installation in virtual environment"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install --require-virtualenv --upgrade ad # (1)!
    ```

    1. You may need to use `pip3` instead of `pip` depending on your Python installation.

    !!! note
        The command to activate the virtual environment depends on your platform and shell.
        [More info](https://docs.python.org/3/library/venv.html#how-venvs-work)

### uv

=== "Adding to uv project"

    ```bash
    uv add ad
    uv sync
    ```

=== "Installing to uv environment"

    ```bash
    uv venv
    uv pip install ad
    ```

### pipenv

```bash
pipenv install ad
```

### poetry

```bash
poetry add ad
```

### pdm

```bash
pdm add ad
```

### hatch

```bash
hatch add ad
```

---

## [GitHub](https://github.com/eggzec/ad)

Install the latest development version directly from the repository:

```bash
pip install --upgrade "git+https://github.com/eggzec/ad.git#egg=ad"
```

### Building locally

Clone and build from source if you want to modify or test local changes:

```bash
git clone https://github.com/eggzec/ad.git
cd ad
pip install -e .
```

---

## Dependencies

- Python 3.10+
- NumPy (optional)
