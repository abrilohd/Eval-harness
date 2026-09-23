import pytest

from evalharness.dataset import load_dataset


def _write(tmp_path, text):
    p = tmp_path / "d.jsonl"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_valid_file(tmp_path):
    p = _write(tmp_path, '{"id":"1","question":"q","expected_sources":["a"]}\n\n')
    assert [c.id for c in load_dataset(p)] == ["1"]


def test_rejects_duplicate_ids(tmp_path):
    line = '{"id":"1","question":"q","expected_sources":["a"]}\n'
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(_write(tmp_path, line * 2))


def test_rejects_empty_expected_sources(tmp_path):
    with pytest.raises(ValueError):
        load_dataset(_write(tmp_path, '{"id":"1","question":"q","expected_sources":[]}\n'))


def test_rejects_empty_file(tmp_path):
    with pytest.raises(ValueError, match="no cases"):
        load_dataset(_write(tmp_path, ""))
