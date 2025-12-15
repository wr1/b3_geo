import yaml

from b3_geo.cli.clean import clean_command


def test_cli_clean(tmp_path, monkeypatch):
    """Test CLI clean command."""
    config_data = {"general": {"workdir": "."}}
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Create the b3_geo dir
    b3_geo_dir = tmp_path / "b3_geo"
    b3_geo_dir.mkdir()
    (b3_geo_dir / "dummy.txt").write_text("dummy")

    # Mock Path.cwd to tmp_path so the relative check passes
    monkeypatch.setattr("b3_geo.cli.clean.Path.cwd", lambda: tmp_path)

    clean_command(str(config_file))

    assert not b3_geo_dir.exists()
