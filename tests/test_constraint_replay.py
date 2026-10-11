"""Invariant: the constraint accepts 100% of training targets.

If this fails, constrained decoding is blocking something the model was trained
to write - it will fight the model at inference and fall back to unconstrained
decoding, which is how an invented price reaches a customer.
"""

from milo.eval.invariants import replay_dataset


def test_constraint_accepts_every_training_target(dataset):
    accepted, total, blocked = replay_dataset(dataset)
    assert blocked == {}, f"{total - accepted}/{total} targets blocked: {blocked}"
    assert accepted == total
