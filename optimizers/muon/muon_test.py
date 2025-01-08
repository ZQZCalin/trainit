"""Tests the muon optimizer."""

import optax
from optimizers.muon.muon import scale_by_muon
from optimizers.optim_test import test_optimizer


def test_muon():
    optimizer = scale_by_muon()

    grad_clip = optax.clip_by_global_norm(10.0)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    test_optimizer(optimizer)


if __name__ == "__main__":
    test_muon()