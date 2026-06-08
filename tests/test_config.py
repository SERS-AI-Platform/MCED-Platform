"""Tests for sers.config — loading, validation, defaults, edge cases."""


import pytest

from sers.config import (
    Config,
    DisplayConfig,
    EquipmentEntry,
    ModelingConfig,
    PreprocessingConfig,
    QCConfig,
    _parse_qc_dict,
    _win_to_wsl_path,
    load_config,
)


# =============================================================================
# QCConfig validation
# =============================================================================
class TestQCConfig:
    """QCConfig dataclass validation."""

    def test_defaults(self):
        qc = QCConfig()
        assert qc.rsd_threshold == 5.0
        assert qc.corr_threshold == 0.95
        assert qc.expected_reps == 5
        assert qc.fingerprint_region == (400.0, 2200.0)
        assert qc.intensity_gate_ratio == 0.1

    def test_custom_values(self):
        qc = QCConfig(rsd_threshold=10.0, corr_threshold=0.90)
        assert qc.rsd_threshold == 10.0
        assert qc.corr_threshold == 0.90

    def test_negative_rsd_raises(self):
        with pytest.raises(ValueError, match="rsd_threshold must be >= 0"):
            QCConfig(rsd_threshold=-1.0)

    def test_zero_rsd_is_valid(self):
        qc = QCConfig(rsd_threshold=0.0)
        assert qc.rsd_threshold == 0.0

    @pytest.mark.parametrize("corr", [0.0, 1.1, -0.5])
    def test_invalid_corr_threshold(self, corr):
        with pytest.raises(ValueError, match="corr_threshold must be in"):
            QCConfig(corr_threshold=corr)

    def test_corr_threshold_boundary(self):
        qc = QCConfig(corr_threshold=1.0)
        assert qc.corr_threshold == 1.0

    def test_inverted_fingerprint_region(self):
        with pytest.raises(ValueError, match="fingerprint_region must be"):
            QCConfig(fingerprint_region=(2200.0, 400.0))

    def test_equal_fingerprint_region(self):
        with pytest.raises(ValueError, match="fingerprint_region must be"):
            QCConfig(fingerprint_region=(1000.0, 1000.0))

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_intensity_gate_ratio(self, ratio):
        with pytest.raises(ValueError, match="intensity_gate_ratio must be in"):
            QCConfig(intensity_gate_ratio=ratio)

    def test_frozen(self):
        qc = QCConfig()
        with pytest.raises(AttributeError):
            qc.rsd_threshold = 10.0


# =============================================================================
# PreprocessingConfig
# =============================================================================
class TestPreprocessingConfig:
    def test_defaults(self):
        p = PreprocessingConfig()
        assert p.do_smooth is True
        assert p.smoothing_method == "savgol"
        assert p.smooth_window == 11
        assert p.smooth_poly == 3
        assert p.median_window == 5
        assert p.gaussian_sigma == 1.0
        assert p.do_baseline is True
        assert p.baseline_method == "rolling_min"
        assert p.baseline_window == 101
        assert p.baseline_airpls_lam == 1e5
        assert p.baseline_arpls_lam == 1e5
        assert p.use_snv is True
        assert p.normalization is None
        assert p.normalization_emsc_order == 2
        assert p.fixed_grid is None

    def test_frozen(self):
        p = PreprocessingConfig()
        with pytest.raises(AttributeError):
            p.do_smooth = False


# =============================================================================
# ModelingConfig
# =============================================================================
class TestModelingConfig:
    def test_defaults(self):
        m = ModelingConfig()
        assert m.n_splits == 5
        assert m.use_pca is False
        assert m.pca_components == 10
        assert m.random_state == 42


# =============================================================================
# DisplayConfig
# =============================================================================
class TestDisplayConfig:
    def test_defaults(self):
        d = DisplayConfig()
        assert d.group_order == []
        assert d.category_map == {}
        assert d.group_colors == {}

    def test_active_groups(self):
        d = DisplayConfig(group_order=["NOR", "DIA", "PRO", "CRC"])
        available = {"CRC", "NOR", "LUN"}
        result = d.active_groups(available)
        assert result == ["NOR", "CRC"]

    def test_palette_for(self):
        d = DisplayConfig(group_colors={"NOR": "#aaa", "CRC": "#bbb"})
        assert d.palette_for(["NOR", "CRC"]) == ["#aaa", "#bbb"]

    def test_palette_for_unknown_group(self):
        d = DisplayConfig(group_colors={"NOR": "#aaa"})
        assert d.palette_for(["NOR", "UNKNOWN"]) == ["#aaa", "#333333"]

    def test_category_for_group(self):
        d = DisplayConfig(category_map={"CRC": "cancer", "NOR": "control"})
        assert d.category_for_group("CRC") == "cancer"
        assert d.category_for_group("UNKNOWN") is None

    def test_category_color(self):
        d = DisplayConfig(category_colors={"cancer": "#E53935"})
        assert d.category_color("cancer") == "#E53935"
        assert d.category_color("unknown") == "#333333"


# =============================================================================
# Config.from_dict
# =============================================================================
class TestConfigFromDict:
    def test_minimal_config(self, minimal_config_dict):
        cfg = Config.from_dict(minimal_config_dict)
        assert cfg.qc.rsd_threshold == 5.0
        assert cfg.preprocessing.smooth_window == 11
        assert cfg.modeling.n_splits == 5
        assert "PRO" in cfg.folder_to_group.values()

    def test_empty_dict_uses_defaults(self):
        cfg = Config.from_dict({})
        assert cfg.qc.rsd_threshold == 5.0
        assert cfg.preprocessing.do_smooth is True
        assert cfg.folder_to_group == {}

    def test_fingerprint_region_list_to_tuple(self):
        d = {"qc": {"fingerprint_region": [500, 1800]}}
        cfg = Config.from_dict(d)
        assert cfg.qc.fingerprint_region == (500, 1800)

    def test_deprecated_qc_fields_ignored(self):
        d = {
            "qc": {
                "rsd_threshold": 5.0,
                "min_snr": 10.0,
                "snr_threshold": 5.0,
                "sbr_threshold": 2.0,
            }
        }
        cfg = Config.from_dict(d)
        assert cfg.qc.rsd_threshold == 5.0

    def test_equipment_mapping(self):
        d = {
            "dataset": {
                "equipment_folder_to_group": {
                    "handheld": {
                        "equipment": "handheld",
                        "sample_folder_to_group": {"1. NOR": "NOR"},
                    }
                }
            }
        }
        cfg = Config.from_dict(d)
        assert "handheld" in cfg.equipment_folder_to_group
        entry = cfg.equipment_folder_to_group["handheld"]
        assert isinstance(entry, EquipmentEntry)
        assert entry.equipment == "handheld"

    def test_display_config(self):
        d = {
            "display": {
                "group_order": ["NOR", "CRC"],
                "group_colors": {"NOR": "#aaa", "CRC": "#bbb"},
                "category_map": {"NOR": "control", "CRC": "cancer"},
            },
            "dataset": {
                "group_metadata": {
                    "NOR": {"label": "Normal"},
                }
            },
        }
        cfg = Config.from_dict(d)
        assert cfg.display.group_order == ["NOR", "CRC"]
        assert cfg.display.group_metadata == {"NOR": {"label": "Normal"}}


# =============================================================================
# load_config from YAML file
# =============================================================================
class TestLoadConfig:
    def test_load_from_file(self, config_yaml_path):
        cfg = load_config(config_yaml_path)
        assert isinstance(cfg, Config)
        assert cfg.qc.rsd_threshold == 5.0

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_empty_yaml(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_config(path)
        assert isinstance(cfg, Config)

    def test_load_accepts_string_path(self, config_yaml_path):
        cfg = load_config(str(config_yaml_path))
        assert isinstance(cfg, Config)


# =============================================================================
# _parse_qc_dict
# =============================================================================
class TestParseQCDict:
    def test_pass_through_current_fields(self):
        raw = {"rsd_threshold": 3.0, "corr_threshold": 0.90}
        result = _parse_qc_dict(raw)
        assert result == {"rsd_threshold": 3.0, "corr_threshold": 0.90}

    def test_deprecated_fields_dropped(self):
        raw = {"rsd_threshold": 5.0, "min_snr": 10.0, "snr_method": "peak"}
        result = _parse_qc_dict(raw)
        assert "min_snr" not in result
        assert "snr_method" not in result
        assert result["rsd_threshold"] == 5.0

    def test_fingerprint_region_list_converted(self):
        raw = {"fingerprint_region": [500, 1800]}
        result = _parse_qc_dict(raw)
        assert result["fingerprint_region"] == (500, 1800)

    def test_fingerprint_region_tuple_kept(self):
        raw = {"fingerprint_region": (500, 1800)}
        result = _parse_qc_dict(raw)
        assert result["fingerprint_region"] == (500, 1800)


# =============================================================================
# WSL path conversion
# =============================================================================
class TestWinToWslPath:
    def test_windows_c_drive(self):
        assert _win_to_wsl_path(r"C:\Users\user\data") == "/mnt/c/Users/user/data"

    def test_windows_d_drive(self):
        assert _win_to_wsl_path(r"D:\Projects\SERS") == "/mnt/d/Projects/SERS"

    def test_already_wsl_path(self):
        assert _win_to_wsl_path("/mnt/c/already/wsl") == "/mnt/c/already/wsl"

    def test_relative_path_unchanged(self):
        assert _win_to_wsl_path("relative/path") == "relative/path"

    def test_backslash_in_relative(self):
        assert _win_to_wsl_path("relative\\path") == "relative/path"

    def test_forward_slash_windows(self):
        assert _win_to_wsl_path("C:/Users/user") == "/mnt/c/Users/user"
