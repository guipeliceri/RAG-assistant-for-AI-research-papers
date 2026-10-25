import pytest

from rag_papers.ids import parse_arxiv_id


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://arxiv.org/abs/2410.12345v2", ("2410.12345", 2)),
        ("2410.12345", ("2410.12345", 1)),
        ("https://arxiv.org/abs/0704.0001v1", ("0704.0001", 1)),
        ("http://arxiv.org/abs/cs/0101001v3", ("cs/0101001", 3)),
        ("http://arxiv.org/abs/math.GT/0309136v1", ("math.GT/0309136", 1)),
    ],
)
def test_parse_arxiv_id(raw, expected):
    assert parse_arxiv_id(raw) == expected


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_arxiv_id("not-an-id")
