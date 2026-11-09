from memory_lab.parsing import parse_json_object


def test_plain_json():
    assert parse_json_object('{"facts": ["a"]}') == {"facts": ["a"]}


def test_fenced_json():
    assert parse_json_object('```json\n{"facts": []}\n```') == {"facts": []}


def test_json_with_chatter_around_it():
    assert parse_json_object('Sure! Here you go: {"facts": ["x"]} Hope that helps.') == {"facts": ["x"]}


def test_garbage_returns_none():
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None
