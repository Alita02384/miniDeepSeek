from __future__ import annotations

from pathlib import Path

from minideepseek.paths import COMPILE_CACHE_ENV, configure_compile_cache


def test_compile_cache_defaults_under_the_cache_root(tmp_path: Path):
    env = {"MINIDEEPSEEK_CACHE_DIR": str(tmp_path)}
    effective = configure_compile_cache(env)

    assert set(effective) == set(COMPILE_CACHE_ENV)
    for variable, directory in COMPILE_CACHE_ENV.items():
        assert effective[variable] == str(tmp_path / directory)
        assert (tmp_path / directory).is_dir()


def test_compile_cache_keeps_explicit_overrides(tmp_path: Path):
    custom = tmp_path / "custom-triton"
    env = {"MINIDEEPSEEK_CACHE_DIR": str(tmp_path), "TRITON_CACHE_DIR": str(custom)}
    effective = configure_compile_cache(env)

    assert effective["TRITON_CACHE_DIR"] == str(custom)
    assert custom.is_dir()
    # Untouched variables still land under the shared root.
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")


def test_compile_cache_is_idempotent(tmp_path: Path):
    env = {"MINIDEEPSEEK_CACHE_DIR": str(tmp_path)}
    assert configure_compile_cache(env) == configure_compile_cache(env)
