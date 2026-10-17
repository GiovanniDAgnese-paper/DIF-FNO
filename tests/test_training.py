"""End-to-end smoke tests for the training loop and the objective."""

from __future__ import annotations

import json

import pytest
import torch

from dagnese_fno.losses import DiffeomorphicOperatorLoss, masked_mse
from dagnese_fno.training import TrainConfig, train_one


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """A 24 x 24, 20-sample dataset: small enough for a unit test, real enough to train."""
    from dagnese_fno.data import generate_dataset, save_dataset

    d = generate_dataset("star", n=24, n_samples=20, seed=0, verbose=False)
    folder = tmp_path_factory.mktemp("data")
    save_dataset(d, folder / "darcy_star.pt")
    return str(folder)


def test_masked_mse_normalisation():
    pred = torch.zeros(1, 8, 8)
    target = torch.ones(1, 8, 8)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:4, :] = True
    by_mask = float(masked_mse(pred, target, mask, normalize_by_mask=True))
    by_total = float(masked_mse(pred, target, mask, normalize_by_mask=False))
    assert by_mask == pytest.approx(1.0)
    assert by_total == pytest.approx(32 / 64)


def test_operator_loss_components():
    pred = torch.randn(2, 16, 16)
    target = torch.randn(2, 16, 16)
    det = torch.full((2, 16, 16), -0.5)      # folded everywhere
    loss = DiffeomorphicOperatorLoss(alpha_h1=0.5, lambda_barrier=2.0, tau=0.05)
    total, comps = loss(pred, target, None, det, spacing=0.1)
    assert set(comps) == {"mse", "h1", "barrier", "total"}
    assert float(comps["barrier"]) > 0       # folding is penalised
    assert float(total) > float(comps["mse"])
    # lambda = 0 -> the barrier disappears from the objective
    loss0 = DiffeomorphicOperatorLoss(lambda_barrier=0.0)
    total0, comps0 = loss0(pred, target, None, det)
    assert float(comps0["barrier"]) == 0.0
    assert float(total0) == pytest.approx(float(comps0["mse"]))


def test_loss_requires_det_when_barrier_is_active():
    loss = DiffeomorphicOperatorLoss(lambda_barrier=1.0)
    with pytest.raises(ValueError):
        loss(torch.randn(1, 8, 8), torch.randn(1, 8, 8), None, None)


@pytest.mark.parametrize("model", ["fno", "dif"])
def test_smoke_training_runs_and_improves(tiny_dataset, model):
    cfg = TrainConfig(domain="star", model=model, epochs=6, batch_size=5,
                      data_dir=tiny_dataset, width=8, modes1=4, modes2=4,
                      fc_channels=16, log_every=100, eval_every=1,
                      lambda_barrier=1.0, out_dir="results")
    result = train_one(cfg, verbose=False)
    assert result["history"], "history must be recorded"
    assert len(result["history"]) == 6
    first, last = result["history"][0], result["history"][-1]
    assert "train_mse" in first and "val_rel_l2" in last
    assert torch.isfinite(torch.tensor(last["val_rel_l2"]))
    assert result["n_train"] == 16 and result["n_val"] == 4
    assert result["n_params"] > 0
    json.dumps(result)          # must be serialisable


def test_diffeomorphic_runs_report_topology(tiny_dataset):
    cfg = TrainConfig(domain="star", model="dif", epochs=3, batch_size=5,
                      data_dir=tiny_dataset, width=8, modes1=4, modes2=4,
                      lambda_barrier=1.0, log_every=100)
    result = train_one(cfg, verbose=False)
    final = result["final"]
    for key in ("min_detJ", "fold_pct", "displacement_mean", "lipschitz_bound",
                "certified_det_lower_bound", "kappa"):
        assert key in final
    assert final["fold_pct"] == 0.0        # zero-init + barrier: no folding this early
    assert final["min_detJ"] > 0


def test_fno_run_has_no_topology(tiny_dataset):
    cfg = TrainConfig(domain="star", model="fno", epochs=2, batch_size=5,
                      data_dir=tiny_dataset, width=8, modes1=4, modes2=4, log_every=100)
    result = train_one(cfg, verbose=False)
    assert result["final"]["min_detJ"] is None
    assert result["final"]["fold_pct"] is None


def test_training_is_deterministic(tiny_dataset):
    kwargs = {"domain": "star", "model": "dif", "epochs": 3, "batch_size": 5, "seed": 3,
                  "data_dir": tiny_dataset, "width": 8, "modes1": 4, "modes2": 4,
                  "lambda_barrier": 1.0, "log_every": 100}
    a = train_one(TrainConfig(**kwargs), verbose=False)
    b = train_one(TrainConfig(**kwargs), verbose=False)
    assert a["final"]["rel_l2"] == pytest.approx(b["final"]["rel_l2"], rel=0, abs=1e-12)


def test_missing_dataset_raises(tmp_path):
    cfg = TrainConfig(domain="star", model="fno", data_dir=str(tmp_path), epochs=1)
    with pytest.raises(FileNotFoundError):
        train_one(cfg, verbose=False)


def test_certified_run_keeps_its_guarantee(tiny_dataset):
    cfg = TrainConfig(domain="star", model="dif", epochs=4, batch_size=5,
                      data_dir=tiny_dataset, width=8, modes1=4, modes2=4,
                      deformation_scale=0.5, lipschitz_mode="spectral",
                      lipschitz_const=0.9, conditioning="global", latent_dim=4,
                      lambda_barrier=1.0, log_every=100)
    result = train_one(cfg, verbose=False)
    final = result["final"]
    assert final["is_certified"] is True
    assert final["certified_det_lower_bound"] > 0
    assert final["min_detJ"] >= final["certified_det_lower_bound"] - 1e-6
    assert final["fold_pct"] == 0.0
