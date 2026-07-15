# Contributing to MCAP Data Loader

Thanks for your interest in improving MCAP Data Loader! This guide covers how to set up a
development environment, the coding conventions we follow, and how to submit changes.

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

MCAP Data Loader targets **Python 3.9+**.

```bash
# 1. Fork and clone
git clone https://github.com/OpenGHz/MCAP-DataLoader.git
cd MCAP-DataLoader

# 2. (Recommended) create an isolated environment
python -m venv .venv && source .venv/bin/activate   # or use conda

# 3. Install the package in editable mode
pip install -e .
```

Optional extras:

```bash
pip install -e ".[letrain]"   # LeRobot training integration (hydra-core, torchdata)
pip install -e ".[nvc]"       # NVIDIA video codec support (PyNvVideoCodec)
```

Some LeRobot-related features additionally require `lerobot` to be installed in your
environment (see the [README](README.md#integration-with-lerobot-training)).

## Code style and quality

This project uses [Ruff](https://docs.astral.sh/ruff/) for formatting and linting, plus a set
of [pre-commit](https://pre-commit.com/) hooks. Install the hooks once and they will run
automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

To run all hooks manually across the codebase:

```bash
pre-commit run --all-files
```

You can also run Ruff directly:

```bash
ruff format .
ruff check --fix .
```

## Tests

Tests live in the [tests](tests) directory and use `pytest`:

```bash
pip install pytest
pytest
```

Please add or update tests when you fix a bug or add a feature, and make sure the suite passes
before opening a pull request.

## Commit messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) style, matching the
existing history. Examples:

- `feat: support LeRobot pi0.5 training with MCAP datasets`
- `fix: correct stale DecodeConfig import path`
- `refactor: depend on cfgable; keep config-framework shims`
- `docs: clarify pi0.5 configuration`

## Pull request process

1. Create a topic branch from `main` (e.g. `feat/my-feature` or `fix/some-bug`).
2. Make your change, keeping it focused and reasonably small.
3. Ensure `pre-commit run --all-files` and `pytest` pass.
4. Update the [README](README.md) / [docs](docs) and [CHANGELOG.md](CHANGELOG.md) when your
   change affects behavior, configuration, or the public API.
5. Open a pull request against `main` with a clear description of what and why. Link any related
   issues.

## Reporting bugs and requesting features

Please use the [issue tracker](https://github.com/OpenGHz/MCAP-DataLoader/issues) and, where
possible, the provided issue templates. For usage questions, see [SUPPORT.md](SUPPORT.md).

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
