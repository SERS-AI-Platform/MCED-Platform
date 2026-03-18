# Changelog

All notable changes to the SERS Analysis Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 

### Changed
- 

### Fixed
- 

---

## [0.1.0] - 2025-01-13

### Added
- Initial release of SERS spectral analysis pipeline
- Preprocessing module with Savitzky-Golay smoothing, baseline correction, SNV normalization
- Quality control with SNR filtering and replicate correlation analysis
- Medoid-based replicate aggregation
- Configuration management with frozen dataclasses
- YAML-based user configuration
- Command-line interface (`sers run`, `sers validate`)
- Docker support for containerized deployment
- Comprehensive test suite with pytest

### Technical Details
- Python 3.10+ required
- Key dependencies: numpy, pandas, scipy, scikit-learn, pyyaml
- Supports environment variable configuration for deployment paths

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2025-01-13 | Initial release |
