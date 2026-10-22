from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arc_agi_trm import build_trm, load_pretrained, load_pretrained_model, save_pretrained
from arc_agi_trm.experiment import load_experiment_config
from arc_agi_trm.operators import Mamba2MLPBlock
from arc_agi_trm.runner import create_experiment_model, create_optimizers
from scripts.train_mamba2_hybrid import _validate_config
from models.losses import ACTLossHead
from test_mamba2_hybrid import FakeMamba2
from test_trm_smoke import synthetic_batch, tiny_config


def _mamba2_mlp_config() -> dict:
    model = tiny_config()
    model.update(
        pos_encodings="none",
        mamba_d_state=4,
        mamba_headdim=8,
        mamba_expand=2,
        mamba_d_conv=4,
        mamba_d_ssm=None,
        mamba_ngroups=1,
        mamba_rmsnorm=True,
        mamba_norm_before_gate=False,
        mamba_chunk_size=16,
        mamba_use_mem_eff_path=True,
        mamba_bias=False,
        mamba_conv_bias=True,
    )
    return model


def _loss_model(config: dict | None = None):
    model = build_trm(
        config or _mamba2_mlp_config(),
        operator="mamba2_mlp",
        mamba_cls=FakeMamba2,
    )
    return ACTLossHead(model, loss_type="stablemax_cross_entropy")


def _artifact_config(model_config: dict | None = None) -> dict:
    return {"operator": "mamba2_mlp", "model": model_config or _mamba2_mlp_config()}


def test_mamba2_mlp_requires_complete_explicit_mamba_configuration() -> None:
    config = tiny_config()
    with pytest.raises(ValueError, match="missing Mamba-2 configuration"):
        build_trm(config, operator="mamba2_mlp", mamba_cls=FakeMamba2)


@pytest.mark.parametrize(
    ("dtype", "forward_dtype"),
    [(torch.float32, "float32"), (torch.bfloat16, "bfloat16")],
)
def test_mamba2_mlp_block_has_mamba_and_swiglu_without_attention_and_backpropagates(
    dtype: torch.dtype, forward_dtype: str
) -> None:
    config = _mamba2_mlp_config()
    config["forward_dtype"] = forward_dtype
    trm = build_trm(
        config, operator="mamba2_mlp", mamba_cls=FakeMamba2
    )
    block = trm.inner.L_level.layers[0]
    assert isinstance(block, Mamba2MLPBlock)
    assert isinstance(block.mamba, FakeMamba2)
    assert not any("Attention" in type(module).__name__ for module in block.modules())

    hidden = torch.randn(2, 11, 32, dtype=dtype, requires_grad=True)
    output = block(cos_sin=None, hidden_states=hidden)
    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype
    assert output.device == hidden.device
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in block.mamba.parameters())
    assert all(parameter.grad is not None for parameter in block.mlp.parameters())


def test_mamba2_mlp_accepts_arc_sequence_length_and_keeps_token_output_shape() -> None:
    config = _mamba2_mlp_config()
    config.update(
        batch_size=1,
        seq_len=900,
        puzzle_emb_len=16,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=1,
    )
    model = ACTLossHead(
        build_trm(config, operator="mamba2_mlp", mamba_cls=FakeMamba2),
        loss_type="stablemax_cross_entropy",
    )
    batch = {
        "inputs": torch.randint(0, 12, (1, 900), dtype=torch.int32),
        "labels": torch.full((1, 900), -100, dtype=torch.int32),
        "puzzle_identifiers": torch.tensor([1], dtype=torch.int32),
    }
    batch["labels"][0, 10] = 3
    carry, loss, _metrics, outputs, _finished = model(
        carry=model.initial_carry(batch),
        batch=batch,
        return_keys=("logits", "preds", "q_halt_logits"),
    )
    assert outputs is not None
    assert outputs["logits"].shape == (1, 900, 12)
    assert outputs["preds"].shape == (1, 900)
    assert carry.inner_carry.z_H.shape == (1, 916, 32)
    assert torch.isfinite(loss)
    loss.backward()
    block = model.model.inner.L_level.layers[0]
    assert all(parameter.grad is not None for parameter in block.mamba.parameters())


def test_mamba2_mlp_preset_and_optimizer_registration(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path(__file__).resolve().parents[1] / "configs" / "mamba2_mlp_100_epochs.json"
    )
    assert config["operator"] == "mamba2_mlp"
    assert _validate_config(config, 1)["per_device_batch_size"] == 32
    config["model"].update(batch_size=2, seq_len=9, vocab_size=12, num_puzzle_identifiers=4)
    metadata = SimpleNamespace(seq_len=9, vocab_size=12, num_puzzle_identifiers=4)
    model = create_experiment_model(
        config,
        metadata,
        per_device_batch_size=2,
        device=torch.device("cpu"),
        mamba_cls=FakeMamba2,
    )
    optimizers, _base_lrs = create_optimizers(
        model,
        {
            "optimization": {
                "puzzle_emb_weight_decay": 0.0,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "puzzle_emb_lr": 0.001,
                "lr": 0.001,
            }
        },
        world_size=1,
        dense_optimizer_cls=torch.optim.AdamW,
    )
    optimized_ids = {
        id(parameter)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    block = model.model.inner.L_level.layers[0]
    assert all(id(parameter) in optimized_ids for parameter in block.mamba.parameters())

    artifact_path = save_pretrained(model, tmp_path / "trainer-model.pt", config)
    restored = create_experiment_model(
        config,
        metadata,
        per_device_batch_size=2,
        device=torch.device("cpu"),
        mamba_cls=FakeMamba2,
    )
    load_pretrained(restored, artifact_path, "cpu", expected_config=config)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_mamba2_mlp_preserves_recursive_trm_interface_and_gradient_boundaries() -> None:
    model = build_trm(
        _mamba2_mlp_config(), operator="mamba2_mlp", mamba_cls=FakeMamba2
    )
    loss_model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    assert not any("Attention" in type(module).__name__ for module in model.modules())
    batch = synthetic_batch()
    grad_modes: list[bool] = []
    handle = model.inner.L_level.register_forward_pre_hook(
        lambda *_args: grad_modes.append(torch.is_grad_enabled())
    )
    try:
        carry, loss, _metrics, outputs, _finished = loss_model(
            carry=loss_model.initial_carry(batch),
            batch=batch,
            return_keys=("logits", "preds", "q_halt_logits"),
        )
    finally:
        handle.remove()

    block = model.inner.L_level.layers[0]
    assert grad_modes == [False] * 6 + [True] * 3
    assert block.mamba.forward_calls == 9
    assert carry.inner_carry.z_H.shape == (2, 11, 32)
    assert carry.inner_carry.z_L.shape == (2, 11, 32)
    assert not carry.inner_carry.z_H.requires_grad
    assert outputs is not None
    assert outputs["logits"].shape == (2, 9, 12)
    assert outputs["preds"].shape == (2, 9)
    assert outputs["q_halt_logits"].shape == (2,)
    loss.backward()
    assert all(parameter.grad is not None for parameter in block.mamba.parameters())
    assert all(parameter.grad is not None for parameter in block.mlp.parameters())


def test_model_only_artifact_roundtrip_reconstructs_and_matches_inference(tmp_path: Path) -> None:
    torch.manual_seed(17)
    source = _loss_model()
    source.eval()
    config = _artifact_config()
    path = save_pretrained(
        source,
        tmp_path / "nested" / "model.pt",
        config,
        metadata={"epoch": 3, "validation_task_exact_match": 0.25},
    )
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["architecture"] == "mamba2_mlp"
    assert payload["vocabulary"] == {"size": 12, "output_size": 12}
    assert payload["architecture_hyperparameters"]["mamba_d_state"] == 4

    restored = load_pretrained_model(path, "cpu", mamba_cls=FakeMamba2)
    assert not restored.training
    for left, right in zip(source.parameters(), restored.parameters()):
        assert torch.equal(left, right)

    batch = synthetic_batch()
    with torch.inference_mode():
        def logits(model):
            carry = model.initial_carry(batch)
            while True:
                carry, _loss, _metrics, output, finished = model(
                    carry=carry,
                    batch=batch,
                    return_keys=("logits", "preds", "q_halt_logits"),
                )
                if bool(finished):
                    return output["logits"]

        assert torch.allclose(logits(source), logits(restored), rtol=0, atol=0)

    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    import_paths = [
        str(repo / "src"),
        str(repo / "external" / "TinyRecursiveModels"),
        str(repo / "tests"),
    ]
    if env.get("PYTHONPATH"):
        import_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(import_paths)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from test_mamba2_hybrid import FakeMamba2; "
            "from arc_agi_trm import load_pretrained_model; "
            "m=load_pretrained_model(sys.argv[1], 'cpu', mamba_cls=FakeMamba2); "
            "assert not m.training",
            str(path),
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_pretrained_loading_rejects_architecture_and_strict_state_mismatches(tmp_path: Path) -> None:
    source = _loss_model()
    path = save_pretrained(source, tmp_path / "model.pt", _artifact_config())

    attention = ACTLossHead(
        build_trm(tiny_config(), operator="attention"),
        loss_type="stablemax_cross_entropy",
    )
    with pytest.raises(ValueError, match="constructed model architecture"):
        load_pretrained(attention, path, "cpu", expected_config={"operator": "attention", "model": tiny_config()})

    malformed = torch.load(path, map_location="cpu", weights_only=True)
    malformed["state_dict"].pop(next(iter(malformed["state_dict"])))
    broken_path = tmp_path / "missing-key.pt"
    torch.save(malformed, broken_path)
    with pytest.raises(ValueError, match="missing_keys"):
        load_pretrained(_loss_model(), broken_path, "cpu", expected_config=_artifact_config())

    target_config = _mamba2_mlp_config()
    target_config["expansion"] = 3.0
    target = _loss_model(target_config)
    with pytest.raises(ValueError, match="configuration_differences"):
        load_pretrained(target, path, "cpu", expected_config=_artifact_config(target_config))


def test_non_strict_pretrained_loading_reports_every_missing_key(tmp_path: Path) -> None:
    source = _loss_model()
    path = save_pretrained(source, tmp_path / "model.pt", _artifact_config())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    removed = next(iter(payload["state_dict"]))
    payload["state_dict"].pop(removed)
    torch.save(payload, path)
    target = _loss_model()
    report = load_pretrained(target, path, "cpu", strict=False, expected_config=_artifact_config())
    assert report["missing_keys"] == [removed]


def test_weights_only_initialization_keeps_fresh_optimizer_state(tmp_path: Path) -> None:
    source = _loss_model()
    path = save_pretrained(source, tmp_path / "model.pt", _artifact_config())
    target = _loss_model()
    optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    assert optimizer.state == {}
    load_pretrained(target, path, "cpu", expected_config=_artifact_config())
    assert optimizer.state == {}
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_only_artifact_can_map_weights_to_cuda(tmp_path: Path) -> None:
    source = _loss_model()
    path = save_pretrained(source, tmp_path / "model.pt", _artifact_config())
    target = _loss_model()
    load_pretrained(target, path, "cuda", expected_config=_artifact_config())
    assert all(parameter.device.type == "cuda" for parameter in target.parameters())
