from memory_lab import data


def test_sample_data_matches_the_notebook():
    convs = data.conversations()
    assert [len(c["messages"]) for c in convs] == [11, 6, 12, 10, 8]
    assert "Portland, Oregon" in convs[2]["messages"][2]["content"]
    assert data.queries()[0] == "What's my name?" and len(data.queries()) == 5
