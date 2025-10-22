import torch

from external.kronos.model import kronos as kronos_module


def test_inference_context_disables_grad() -> None:
    prev_state = torch.is_grad_enabled()
    try:
        with kronos_module._inference_context():
            assert not torch.is_grad_enabled()
    finally:
        torch.set_grad_enabled(prev_state)


def test_fast_settings_singleton() -> None:
    original_flag = kronos_module._FAST_TORCH_SETTINGS_CONFIGURED
    try:
        kronos_module._FAST_TORCH_SETTINGS_CONFIGURED = False
        kronos_module._maybe_enable_fast_torch_settings()
        assert kronos_module._FAST_TORCH_SETTINGS_CONFIGURED is True
        kronos_module._maybe_enable_fast_torch_settings()
        assert kronos_module._FAST_TORCH_SETTINGS_CONFIGURED is True
    finally:
        kronos_module._FAST_TORCH_SETTINGS_CONFIGURED = original_flag
