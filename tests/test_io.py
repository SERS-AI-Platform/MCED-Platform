"""Tests for sers.io — filename parsing, reading, finding, grid creation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sers.io import (
    DatasetResult,
    SpectrumID,
    find_spectra,
    load_dataset,
    make_common_grid,
    make_fixed_grid,
    parse_filename,
    read_spectrum,
)


# =============================================================================
# parse_filename
# =============================================================================
class TestParseFilename:
    """Test filename parsing patterns."""

    def test_space_separator(self):
        result = parse_filename(Path("CRC 001_1.csv"))
        assert result == SpectrumID("CRC", "001", 1)

    def test_space_separator_uppercase(self):
        result = parse_filename(Path("CRC 042_3.CSV"))
        assert result == SpectrumID("CRC", "042", 3)

    def test_dotted_group(self):
        result = parse_filename(Path("H.D. 042_3.csv"))
        assert result == SpectrumID("H.D.", "042", 3)

    def test_dotted_group_underscore(self):
        result = parse_filename(Path("H.D._042_3.csv"))
        assert result == SpectrumID("H.D.", "042", 3)

    def test_simple_group(self):
        result = parse_filename(Path("PRO 100_5.csv"))
        assert result == SpectrumID("PRO", "100", 5)

    def test_nor_group(self):
        result = parse_filename(Path("NOR 001_1.csv"))
        assert result == SpectrumID("NOR", "001", 1)

    def test_fallback_pattern(self):
        result = parse_filename(Path("001_3.csv"), fallback_group="LUN")
        assert result == SpectrumID("LUN", "001", 3)

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Cannot parse filename"):
            parse_filename(Path("completely_invalid.csv"))

    def test_lowercase_group_uppercased(self):
        result = parse_filename(Path("crc 001_1.csv"))
        assert result.group == "CRC"

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("BLC 001_1.csv", SpectrumID("BLC", "001", 1)),
            ("OVA 070_5.csv", SpectrumID("OVA", "070", 5)),
            ("CPAN 001_1.csv", SpectrumID("CPAN", "001", 1)),
            ("SPAN 072_3.csv", SpectrumID("SPAN", "072", 3)),
        ],
    )
    def test_various_groups(self, filename, expected):
        result = parse_filename(Path(filename))
        assert result == expected


# =============================================================================
# read_spectrum
# =============================================================================
class TestReadSpectrum:
    def test_reads_csv(self, tmp_spectrum_csv):
        x, y = read_spectrum(tmp_spectrum_csv)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == len(y)
        assert len(x) > 0

    def test_sorted_by_raman_shift(self, tmp_path):
        # Write reversed data
        x = np.array([2000.0, 1500.0, 1000.0, 500.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])
        df = pd.DataFrame({"x": x, "y": y})
        path = tmp_path / "reversed.csv"
        df.to_csv(path, index=False, header=False)

        x_read, y_read = read_spectrum(path)
        assert x_read[0] < x_read[-1]  # Should be sorted ascending
        np.testing.assert_array_equal(x_read, [500.0, 1000.0, 1500.0, 2000.0])
        np.testing.assert_array_equal(y_read, [1.0, 2.0, 3.0, 4.0])

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            read_spectrum(tmp_path / "does_not_exist.csv")

    def test_tab_separated(self, tmp_path):
        x = np.linspace(400, 2200, 50)
        y = np.random.default_rng(0).random(50) * 1000
        lines = [f"{xi}\t{yi}" for xi, yi in zip(x, y)]
        path = tmp_path / "tab_sep.txt"
        path.write_text("\n".join(lines))

        x_read, y_read = read_spectrum(path)
        assert len(x_read) == 50


# =============================================================================
# find_spectra
# =============================================================================
class TestFindSpectra:
    def test_finds_csv_files(self, tmp_spectrum_dir):
        group_dir = list(tmp_spectrum_dir.iterdir())[0]
        files = find_spectra(group_dir, pattern="*.csv", recursive=False)
        assert len(files) == 9  # 3 samples x 3 reps

    def test_sorted_output(self, tmp_spectrum_dir):
        group_dir = list(tmp_spectrum_dir.iterdir())[0]
        files = find_spectra(group_dir)
        assert files == sorted(files)

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        files = find_spectra(empty)
        assert files == []

    def test_non_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.csv").write_text("1,2\n3,4")
        (sub / "nested.csv").write_text("1,2\n3,4")
        files = find_spectra(tmp_path, recursive=False)
        assert len(files) == 1


# =============================================================================
# make_common_grid
# =============================================================================
class TestMakeCommonGrid:
    def test_basic(self):
        arrays = [
            np.linspace(400, 2200, 1000),
            np.linspace(380, 2250, 1100),
            np.linspace(420, 2180, 950),
        ]
        grid = make_common_grid(arrays)
        # Should use intersection: max of mins, min of maxs
        assert grid[0] >= 420.0
        assert grid[-1] <= 2180.0

    def test_custom_bounds(self):
        arrays = [np.linspace(400, 2200, 1000)]
        grid = make_common_grid(arrays, x_min=500, x_max=2000, n_points=100)
        assert len(grid) == 100
        np.testing.assert_allclose(grid[0], 500.0)
        np.testing.assert_allclose(grid[-1], 2000.0)

    def test_n_points_defaults_to_median(self):
        arrays = [
            np.linspace(400, 2200, 100),
            np.linspace(400, 2200, 200),
            np.linspace(400, 2200, 300),
        ]
        grid = make_common_grid(arrays)
        assert len(grid) == 200  # median of [100, 200, 300]


# =============================================================================
# make_fixed_grid
# =============================================================================
class TestMakeFixedGrid:
    def test_with_fixed_grid_config(self):
        from sers.config import Config, PreprocessingConfig

        prep = PreprocessingConfig(
            fixed_grid={"x_min": 402.0, "x_max": 2198.0, "n_points": 935}
        )
        cfg = Config(preprocessing=prep)
        grid = make_fixed_grid(cfg)
        assert len(grid) == 935
        np.testing.assert_allclose(grid[0], 402.0)
        np.testing.assert_allclose(grid[-1], 2198.0)

    def test_without_fixed_grid_returns_none(self):
        from sers.config import Config

        cfg = Config()
        grid = make_fixed_grid(cfg)
        assert grid is None


# =============================================================================
# DatasetResult
# =============================================================================
class TestDatasetResult:
    def test_len(self):
        dr = DatasetResult(spectra={("A", "1", 1): (np.array([1]), np.array([2]))},
                           metadata=pd.DataFrame(), failed_files=[])
        assert len(dr) == 1

    def test_groups_from_metadata(self):
        meta = pd.DataFrame([
            {"group": "CRC", "sample_id": "001", "replicate": 1},
            {"group": "NOR", "sample_id": "001", "replicate": 1},
        ])
        dr = DatasetResult(spectra={}, metadata=meta, failed_files=[])
        assert dr.groups() == ["CRC", "NOR"]

    def test_groups_empty_metadata(self):
        dr = DatasetResult(spectra={}, metadata=pd.DataFrame(), failed_files=[])
        assert dr.groups() == []

    def test_has_equipment(self):
        meta = pd.DataFrame([{"equipment": "handheld", "group": "NOR"}])
        dr = DatasetResult(spectra={}, metadata=meta, failed_files=[])
        assert dr.has_equipment is True

    def test_no_equipment(self):
        meta = pd.DataFrame([{"group": "NOR"}])
        dr = DatasetResult(spectra={}, metadata=meta, failed_files=[])
        assert dr.has_equipment is False


# =============================================================================
# load_dataset
# =============================================================================
class TestLoadDataset:
    def test_flat_mode(self, tmp_spectrum_dir):
        folder_to_group = {"1. CRC (10)": "CRC"}
        result = load_dataset(
            tmp_spectrum_dir,
            folder_to_group=folder_to_group,
            show_progress=False,
        )
        assert isinstance(result, DatasetResult)
        assert len(result) == 9
        assert result.groups() == ["CRC"]
        assert len(result.failed_files) == 0

    def test_must_provide_exactly_one_mapping(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one"):
            load_dataset(tmp_path)

        with pytest.raises(ValueError, match="exactly one"):
            load_dataset(
                tmp_path,
                folder_to_group={"a": "A"},
                equipment_mapping={"b": None},
            )

    def test_group_filter(self, tmp_spectrum_dir):
        folder_to_group = {"1. CRC (10)": "CRC"}
        # Filter for a group that doesn't exist → empty
        result = load_dataset(
            tmp_spectrum_dir,
            folder_to_group=folder_to_group,
            groups=["NOR"],
            show_progress=False,
        )
        assert len(result) == 0

    @pytest.mark.slow
    def test_metadata_columns(self, tmp_spectrum_dir):
        folder_to_group = {"1. CRC (10)": "CRC"}
        result = load_dataset(
            tmp_spectrum_dir,
            folder_to_group=folder_to_group,
            show_progress=False,
        )
        expected_cols = {"file", "folder", "group", "sample_id", "replicate",
                         "n_points", "x_min", "x_max", "y_min", "y_max"}
        assert expected_cols.issubset(set(result.metadata.columns))


# =============================================================================
# SpectrumID
# =============================================================================
class TestSpectrumID:
    def test_named_tuple(self):
        sid = SpectrumID("CRC", "001", 1)
        assert sid.group == "CRC"
        assert sid.sample_id == "001"
        assert sid.replicate == 1

    def test_equality(self):
        a = SpectrumID("CRC", "001", 1)
        b = SpectrumID("CRC", "001", 1)
        assert a == b

    def test_unpacking(self):
        sid = SpectrumID("PRO", "042", 3)
        group, sample_id, rep = sid
        assert group == "PRO"
        assert sample_id == "042"
        assert rep == 3
