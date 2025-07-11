# Python Package Template

[![](https://img.shields.io/badge/Python-3.10|3.11|3.12-blue)](https://www.python.org)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/Pyright-enabled-brightgreen)](https://github.com/microsoft/pyright)
[![codecov](https://codecov.io/gh/michaelellis003/python-package-template/graph/badge.svg?token=TUKP19SKJ3)](https://codecov.io/gh/michaelellis003/python-package-template)
[![License](https://img.shields.io/github/license/michaelellis003/python-package-template)](https://github.com/michaelellis003/python-package-template/blob/main/LICENSE)

The `python-package-template` repository offers a robust template for creating Python packages. It incorporates best practices for project structure, dependency management, testing, and CI/CD, enabling developers to quickly set up and maintain high-quality Python projects.

## Table of Contents
1. [Features](#features)
2. [How to use](#how-to-use)
3. [Using Poetry](#using-poetry)
   - [Managing Dependencies](#managing-dependencies)
   - [Updating Package Version](#updating-package-version)
5. [CI/CD Workflows](#ci-cd-workflows)

## Features
- [Poetry](https://python-poetry.org) for Python package management and environment handling.
- [Pre-commit hooks](https://pre-commit.com) to enforce consistent code style, including:
    - [Ruff](https://docs.astral.sh/ruff/) for linting and formatting,
    - [Pyright](https://github.com/microsoft/pyright) for static type checking.
- [Pytest](https://docs.pytest.org/en/stable/) for running code tests.
- **GitHub Actions** for CI/CD, including automated tests, lint checks, and release tagging.

## How to use
1. Install Poetry
- [See the Poetry documentation](https://python-poetry.org) for more details and alternate methods. Examples include:
    ```
    # using pipx.
    pipx install poetry
    ```

2. Ensure virtual enivornment is intalled in your project directory
    ```
    poetry config virtualenvs.in-project true`
    ```

2. Install package dependencies needed for development:
    ```
    poetry install
    ```

3. Enable pre-commit hooks in your local environment so they run automatically before every commit:
    ```
    poetry run pre-commit install
    ```

4. Update project metadata in:
    - `pyproject.toml`:
        - Change the project name, version, and author information to match your package.
    - `README.md`, `LICENSE`, `CHANGELOG.md` (optional):
        - Replace placeholder names, badges, and repository links with those for your project.

## Using Poetry
For the full Poetry documentation, visit the [full docs](https://python-poetry.org)

### Managing Dependencies
- Adding Dependencies
    - To add a new runtime dependency to your project, use:
        ```
        poetry add <package_name>
        ```
    - Example:
        ```
        poetry add requests
        ```
        This updates your `pyproject.toml` under [project.dependencies] and synchronizes your virtual environment automatically.

    - For dev-only dependencies, you can specify --dev:
        ```
        poetry add pytest --group dev
        ```
        This updates [tool.poetry.group.dev.dependencies] in your pyproject.toml.

    - Poetry provides a way to organize your dependencies by groups. So you can
    create a new dependency group:
        ```
        poetry add pytest --group <new-dependency-group>
        ```
        Read more about this [here](https://python-poetry.org/docs/managing-dependencies/)

- Removing Dependencies
    - Similarly, to remove a dependency:
        ```
        poetry remove requests
        ```
        Poetry removes the package from your pyproject.toml and uninstalls it from your virtual environment.

### Updating Package Version
- Before merging a branch into main to release a new version of your package you will need to update the version number in the pyproject.toml. If you do not update the verrsion number before merging to the main branch the release-and-tag.yml workflow will fail.
    ```
    poetry version <bump-rule>
    ```
    Provide a valid bump rule: patch, minor, major, prepatch, preminor, premajor, prerelease.

## CI-CD Workflows

This project uses GitHub Actions for continuous integration and deployment.

### On Push to Non-Main Branches

- **Linting & Formatting:** Runs `pre-commit` checks using `ruff`.
- **Testing:** Runs `pytest` across Python 3.10, 3.11, and 3.12.
- **Coverage Upload:** Sends test coverage reports to Codecov.

### On Merging into Main

- **Tagging & Releasing:** Automatically tags a new version based on `pyproject.toml`.
- **Builds the Package:** Uses Poetry to create distribution files.
- **Creates a GitHub Release:** Uploads the built package to GitHub releases.