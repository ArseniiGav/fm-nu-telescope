import subprocess
import sys

import numpy as np
import pytest

from nutel.toy.cli import _overrides, build_parser, default_output, main
from nutel.toy.config import config_fields, preset
from nutel.toy.io import describe, load, save
from synthetic import synthetic_data, two_detectors


@pytest.fixture
def files(tmp_path):
    a = save(synthetic_data(6, seed=1), tmp_path / "a.h5")
    b = save(synthetic_data(5, seed=2, detectors=two_detectors(), first_event_id=6), tmp_path / "b.h5")
    return a, b


def test_help_lists_every_parameter(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["generate", "--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for _, f in config_fields():
        assert "--" + f.name.replace("_", "-") in text, f.name
    for flag in ("--no-scattering", "--fixed-direction", "--n-geometries", "--level", "--save-paths", "--family"):
        assert flag in text
    for title in ("multi-detector data sets", "fixed event parameters", "medium", "readout", "advanced options"):
        assert title in text
    for command in ("info", "merge", "presets", "family", "noise", "display", "animate"):
        with pytest.raises(SystemExit):
            main([command, "--help"])


def test_parser_overrides():
    args = build_parser().parse_args(
        ["generate", "--no-scattering", "--window", "0", "3000", "--event-type", "cascade", "--n-strings", "5"]
    )
    assert args.scattering is False
    assert args.absorption is None  # unchanged: the preset's value
    assert args.window == [0.0, 3000.0]
    assert args.event_type == "cascade"
    assert args.n_strings == 5


def test_fixed_direction_to_angle():
    def angle(zenith, azimuth):
        args = build_parser().parse_args(["generate", "--fixed-direction", str(zenith), str(azimuth)])
        return _overrides(args)["fixed_angle"]

    assert angle(0, 0) == pytest.approx(-90.0)  # from straight above: travels down
    assert abs(angle(90, 0)) == pytest.approx(180.0)  # horizontal from +x: travels towards -x
    assert angle(90, 180) == pytest.approx(0.0, abs=1e-12)  # horizontal from -x: travels towards +x
    assert angle(180, 0) == pytest.approx(90.0)  # up-going
    args = build_parser().parse_args(["generate", "--fixed-direction", "10", "0", "--fixed-angle", "5"])
    with pytest.raises(ValueError, match="not both"):
        _overrides(args)


def test_default_output_name():
    assert default_output("full", preset("full"), 1000, 3).as_posix() == "data/toy_full_water_n1000_seed3.h5"
    ice = preset("scatter", medium="ice")
    assert default_output("scatter", ice, 10, 0, medium="ice").name == "toy_scatter_ice_n10_seed0.h5"
    custom = default_output("full", preset("full", noise_rate=0.0), 10, 0)
    assert custom.name.startswith("toy_full-custom-")
    family = default_output("full", preset("full"), 10, 0, family="default", n_geometries=4)
    assert family.name == "toy_full_default-k4_n10_seed0.h5"


def test_presets(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    for name in ("straight", "absorb", "scatter", "full", "water", "ice"):
        assert name in out


def test_info_and_merge(files, tmp_path, capsys):
    a, b = files
    assert main(["info", str(b)]) == 0
    out = capsys.readouterr().out
    assert "5 events, 2 detector(s)" in out
    assert main(["merge", "-q", "-o", str(tmp_path / "ab.h5"), str(a), str(b)]) == 0
    assert len(load(tmp_path / "ab.h5")) == 11
    assert main(["merge", "-q", "-o", str(tmp_path / "aa.h5"), str(a), str(a)]) == 1
    assert "seed" in capsys.readouterr().err
    assert main(["info", str(tmp_path / "missing.h5")]) == 1


def test_display_and_animate(files, tmp_path):
    a, b = files
    assert main(["display", str(b), "-i", "1", "-q"]) == 0
    assert (tmp_path / "b_event1.png").stat().st_size > 1000
    out = tmp_path / "fig" / "paths.pdf"
    assert main(["display", str(a), "-i", "0", "--paths", "--no-linefit", "-o", str(out), "-q"]) == 0
    assert out.exists()
    assert main(["display", str(a), "-i", "0", "--level", "hits", "-o", str(tmp_path / "h.png"), "-q"]) == 0
    assert main(["display", str(a), "-i", "99", "-o", str(tmp_path / "x.png"), "-q"]) == 1
    gif = tmp_path / "a.gif"
    assert main(["animate", str(a), "-i", "2", "--frames", "3", "--dpi", "40", "-o", str(gif), "-q"]) == 0
    assert gif.stat().st_size > 1000


def test_family(tmp_path, capsys):
    out = tmp_path / "family.png"
    assert main(["family", "--n", "3", "--seed", "2", "-o", str(out), "-q"]) == 0
    text = capsys.readouterr().out
    assert "[2]" in text
    assert out.stat().st_size > 1000
    with pytest.raises(SystemExit) as error:
        main(["family", "--family", "nonexistent", "--n", "2", "-q"])
    assert error.value.code == 2
    assert "invalid choice: 'nonexistent'" in capsys.readouterr().err


def test_family_refuses_options_it_would_ignore(tmp_path, capsys):
    argv = ["generate", "--family", "default", "--n-geometries", "2", "-n", "2", "--device", "cpu", "-q"]
    argv += ["--medium", "ice", "--noise-rate", "0", "--layout", "core", "--efficiency", "0.3"]
    with pytest.raises(SystemExit) as error:
        main([*argv, "-o", str(tmp_path / "x.h5")])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert "--medium, --layout, --efficiency, --noise-rate" in err
    assert "samples the layout, the medium and the readout efficiency" in err
    assert not (tmp_path / "x.h5").exists()
    with pytest.raises(SystemExit):
        main(["generate", "--family", "nope", "-n", "2", "-q"])
    assert "invalid choice: 'nope'" in capsys.readouterr().err


def test_family_file_describes_sampled_parts(tmp_path):
    out = tmp_path / "fam.h5"
    argv = ["generate", "--family", "default", "--n-geometries", "2", "-n", "4", "--preset", "straight"]
    with pytest.raises(SystemExit):
        main([*argv, "--time-jitter", "0", "-q", "-o", str(out)])
    argv += ["--merge-window", "5", "--pulse-time", "mean", "--device", "cpu", "-q", "-o", str(out)]
    assert main(argv) == 0
    data = load(out)
    assert data.config.readout.merge_window == 5
    assert data.config.readout.pulse_time == "mean"
    assert all(d.readout.merge_window == 5 for d in data.detectors)
    assert data.provenance["medium"] == "sampled per detector"
    text = describe(out)
    assert "detectors sampled from the family default" in text
    assert "layout:     sampled per detector" in text
    assert "medium:     sampled per detector; absorption off; scattering off" in text
    assert "readout:    efficiency, jitter and noise sampled per detector, window" in text
    assert "medium:     water" not in text


def test_noise_augmentation(files, tmp_path):
    a, _ = files
    out = tmp_path / "noisy.h5"
    assert main(["noise", str(a), "--rate", "20000", "--seed", "3", "-o", str(out), "-q"]) == 0
    before, after = load(a), load(out)
    assert after.photons.n_noise.sum() > before.photons.n_noise.sum()
    assert main(["noise", str(a), "--rate", "20000", "-o", str(out), "-q"]) == 1  # exists
    no_hits = save(synthetic_data(3, hits=False), tmp_path / "p.h5")
    assert main(["noise", str(no_hits), "--rate", "1000", "-q"]) == 1


def test_generate_overrides_reach_the_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = ["generate", "--preset", "straight", "--medium", "ice", "--device", "cpu", "-n", "6", "--seed", "4"]
    argv += ["--event-type", "cascade", "--fixed-energy", "50", "--noise-rate", "1000", "--save-paths", "1"]
    argv += ["--no-progress", "-q"]
    assert main(argv) == 0
    expected = preset("straight", medium="ice", event_type="cascade", fixed_energy=50.0, noise_rate=1000.0)
    out = default_output("straight", expected, 6, 4, medium="ice")
    assert "straight-custom-" in out.name
    assert "_ice_" in out.name
    data = load(out)
    assert data.config == expected
    assert len(data) == 6
    assert np.all(data.truth.type == 2)
    np.testing.assert_allclose(data.truth.energy, 50.0)
    assert data.hits is not None
    assert data.paths is not None
    assert data.provenance["command_line"].startswith("nutelsim generate --preset straight")
    assert main(argv) == 1  # the file exists


def test_generate_multi_detector_end_to_end(tmp_path):
    common = [sys.executable, "-m", "nutel.toy", "generate", "--device", "cpu", "--preset", "straight", "-q"]
    common += ["--family", "default", "--n-geometries", "3", "--events-per-geometry", "2", "--level", "pulses"]
    a = subprocess.run([*common, "-o", str(tmp_path / "a.h5")], capture_output=True, text=True, check=False)
    assert a.returncode == 0, a.stderr
    b = subprocess.run(
        [*common, "--seed", "1", "--first-event-id", "6", "-o", str(tmp_path / "b.h5")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert b.returncode == 0, b.stderr
    data = load(tmp_path / "a.h5")
    assert data.n_geometries == 3
    assert len(data) == 6
    assert data.hits is None
    np.testing.assert_array_equal(np.bincount(data.truth.geometry_index), [2, 2, 2])
    assert main(["merge", "-q", "-o", str(tmp_path / "ab.h5"), str(tmp_path / "a.h5"), str(tmp_path / "b.h5")]) == 0
    assert load(tmp_path / "ab.h5").n_geometries == 3  # the same detectors (same --geometry-seed) stored once
