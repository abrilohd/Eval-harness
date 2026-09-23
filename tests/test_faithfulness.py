from evalharness.metrics.faithfulness import parse_judgement


def test_parses_plain_json():
    assert parse_judgement('{"score": 0.5, "unsupported": ["x"]}') == (0.5, ["x"])


def test_parses_fenced_json():
    text = '```json\n{"score": 1, "unsupported": []}\n```'
    assert parse_judgement(text) == (1.0, [])


def test_clamps_score():
    assert parse_judgement('{"score": 3}')[0] == 1.0
    assert parse_judgement('{"score": -1}')[0] == 0.0
