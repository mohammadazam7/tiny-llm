import math

import pytest
import torch

from tiny_llm.model import ConfigError, ModelConfig, TinyLM

SMALL = ModelConfig(vocab_size=50, context=16, layers=2, heads=4, width=32)


def tokens(batch: int, length: int, config: ModelConfig = SMALL, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, config.vocab_size, (batch, length), generator=generator)


def model(config: ModelConfig = SMALL, seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(config).eval()


def test_logits_have_one_score_per_vocabulary_entry_at_every_position() -> None:
    assert model()(tokens(3, 11)).shape == (3, 11, 50)


def test_parameter_count_matches_the_hand_calculation() -> None:
    # Tokens 50*32 = 1,600 and positions 16*32 = 512. One block: 12*32*32 + 13*32 = 12,704.
    # Final norm 2*32 = 64. Total 1,600 + 512 + 2*12,704 + 64 = 27,584.
    assert SMALL.parameter_count() == 27_584
    assert model().parameter_count() == 27_584


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(vocab_size=512, context=64, layers=1, heads=1, width=16),
        ModelConfig(vocab_size=300, context=32, layers=3, heads=2, width=24),
        ModelConfig(vocab_size=1024, context=128, layers=4, heads=4, width=128),
    ],
)
def test_parameter_formula_holds_for_other_sizes(config: ModelConfig) -> None:
    assert TinyLM(config).parameter_count() == config.parameter_count()


@pytest.mark.parametrize("changed", [1, 5, 10])
def test_a_later_token_cannot_change_an_earlier_prediction(changed: int) -> None:
    net = model()
    a = tokens(2, 11)
    b = a.clone()
    b[:, changed] = (b[:, changed] + 1) % SMALL.vocab_size
    out_a, out_b = net(a), net(b)
    # Positions before the change see identical input, so their logits are identical.
    assert torch.equal(out_a[:, :changed], out_b[:, :changed])
    # The changed position itself must differ, or the test would prove nothing.
    assert not torch.allclose(out_a[:, changed], out_b[:, changed])


def test_the_prefix_of_a_sequence_gets_the_same_logits_as_the_prefix_alone() -> None:
    net = model()
    full = tokens(1, 12)
    assert torch.allclose(net(full)[:, :7], net(full[:, :7]), atol=1e-6)


def test_attention_never_looks_ahead_and_each_row_sums_to_one() -> None:
    maps = model().attention(tokens(2, 9))
    assert len(maps) == SMALL.layers
    future = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)
    for weights in maps:
        assert weights.shape == (2, SMALL.heads, 9, 9)
        assert torch.all(weights[..., future] == 0)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, SMALL.heads, 9), atol=1e-6)
    # The first position can only look at itself.
    assert torch.all(maps[0][..., 0, 0] == 1)


def test_the_output_head_shares_the_token_embedding() -> None:
    net = model()
    assert net.head.weight is net.token_embedding.weight


def test_loss_at_initialisation_is_close_to_a_uniform_guess() -> None:
    net = model()
    data = tokens(8, 16, seed=1)
    loss = net.loss(data[:, :-1], data[:, 1:]).item()
    assert loss == pytest.approx(math.log(SMALL.vocab_size), abs=0.1)


def test_every_parameter_receives_a_gradient() -> None:
    net = model()
    data = tokens(4, 16, seed=2)
    net.loss(data[:, :-1], data[:, 1:]).backward()
    missing = [name for name, p in net.named_parameters() if p.grad is None or not p.grad.any()]
    assert missing == []


def test_one_batch_can_be_memorised() -> None:
    torch.manual_seed(0)
    net = TinyLM(SMALL).train()
    data = tokens(4, 16, seed=3)
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-3)
    first = net.loss(data[:, :-1], data[:, 1:]).item()
    for _ in range(200):
        optimiser.zero_grad()
        loss = net.loss(data[:, :-1], data[:, 1:])
        loss.backward()
        optimiser.step()
    assert loss.item() < 0.1 < first


def test_dropout_is_off_in_evaluation_and_on_in_training() -> None:
    config = ModelConfig(vocab_size=50, context=16, layers=2, heads=4, width=32, dropout=0.5)
    net = model(config)
    data = tokens(2, 10)
    assert torch.equal(net(data), net(data))
    net.train()
    assert not torch.equal(net(data), net(data))


def test_a_sequence_longer_than_the_context_is_refused() -> None:
    with pytest.raises(ValueError, match="longer than the context of 16"):
        model()(tokens(1, 17))


def test_a_flat_list_of_tokens_is_refused() -> None:
    with pytest.raises(ValueError, match="batch, tokens"):
        model()(torch.zeros(5, dtype=torch.long))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vocab_size": 50, "width": 30, "heads": 4}, "not divisible"),
        ({"vocab_size": 0}, "vocab_size must be at least 1"),
        ({"vocab_size": 50, "layers": 0}, "layers must be at least 1"),
        ({"vocab_size": 50, "dropout": 1.0}, "dropout"),
    ],
)
def test_impossible_configurations_are_refused(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        ModelConfig(**kwargs)
