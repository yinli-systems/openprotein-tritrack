import torch

from openprotein.attention import build_bidirectional_mask, build_prefix_lm_mask


def test_bidirectional_mask_respects_padding():
    protein = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    text = torch.tensor([[1, 0]], dtype=torch.bool)
    actual = build_bidirectional_mask(protein, text)[0]
    valid = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
    assert torch.equal(actual, valid[:, None] & valid[None, :])


def test_prefix_lm_mask_prevents_target_leakage():
    protein = torch.ones((1, 2), dtype=torch.bool)
    text = torch.ones((1, 3), dtype=torch.bool)
    actual = build_prefix_lm_mask(protein, text)[0]
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(actual, expected)


def test_prefix_lm_mask_removes_padded_queries_and_keys():
    protein = torch.tensor([[1, 0]], dtype=torch.bool)
    text = torch.tensor([[1, 0]], dtype=torch.bool)
    actual = build_prefix_lm_mask(protein, text)[0]
    assert not actual[1].any()
    assert not actual[3].any()
    assert not actual[:, 1].any()
    assert not actual[:, 3].any()
