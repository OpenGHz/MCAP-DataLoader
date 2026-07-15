# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Community and repository documentation: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `SUPPORT.md`, `CITATION.cff`, GitHub issue/PR templates, and a Simplified Chinese README
  (`README.zh-CN.md`).

## [0.3.2]

### Added

- LeRobot **pi0.5** (vision-language-action) training support with MCAP datasets, including
  per-episode language-task extraction from MCAP metadata and quantile (`q01`/`q99`) statistics.

### Changed

- Compatibility with LeRobot 0.6.0 for the pi0.5 training entry point.
- Refined project dependencies in `pyproject.toml`.
- Renamed the console command from `mcap_lerobot_train` to `mcap-lerobot-train` (documentation
  and scripts updated accordingly).

### Fixed

- Avoided video decoding during the pi0.5 quantile pass to prevent a DataLoader worker deadlock.

## [0.3.1]

Baseline release published to [PyPI](https://pypi.org/project/mcap-data-loader/).

---

For the complete history prior to this changelog, see the
[commit log](https://github.com/OpenGHz/MCAP-DataLoader/commits/main).

[Unreleased]: https://github.com/OpenGHz/MCAP-DataLoader/compare/main...HEAD
