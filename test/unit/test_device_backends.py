"""Backend routing tests use mocks; no NPU tensors or driver are required."""
import argparse
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import torch

from training.utils import device as devices
from training.utils.checkpoint import save_checkpoint

TORCH_DEVICE = torch.device
ROOT = Path(__file__).resolve().parents[2]


def fake_device(index=0):
    return cast("torch.device", SimpleNamespace(type="npu", index=index))


def fake_npu():
    generator = torch.Generator().manual_seed(17)
    backend = Mock()
    backend.is_available.return_value = True
    backend.device_count.return_value = 4
    backend.get_device_name.return_value = "Mock Ascend"
    backend.manual_seed_all.side_effect = generator.manual_seed
    backend.get_rng_state_all.side_effect = lambda: [generator.get_state()]
    backend.set_rng_state_all.side_effect = lambda states: generator.set_state(states[0])
    return backend, generator


def device_factory(name):
    if str(name).startswith("npu:"):
        return fake_device(int(str(name).split(":")[1]))
    return TORCH_DEVICE(name)


class SimulatedAcceleratorTensor(torch.Tensor):
    """CPU-backed test data that requires conversion before safe serialization."""
    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, torch.as_tensor(data, dtype=torch.float32), False)

    @property
    def device(self) -> "torch.device":
        return fake_device()

    def cpu(self):
        return self.as_subclass(torch.Tensor).detach().clone()


class DeviceBackendTests(unittest.TestCase):
    def setUp(self):
        state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, state)

    def test_argument_helpers_remain_usable_without_importing_torch(self):
        code = '''import argparse, sys
from training.utils.device import add_device_argument
parser = argparse.ArgumentParser()
add_device_argument(parser)
assert parser.parse_args(["--device", "npu:2"]).device == "npu:2"
assert "torch" not in sys.modules
assert "torch_npu" not in sys.modules
from training.run_game_ai import parse_args
assert parse_args(["--device", "npu:1"]).device == "npu:1"
assert "torch" not in sys.modules
'''
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)

    def test_all_training_and_testing_entries_accept_indexed_npu(self):
        from training import DQN_train, DRQN_train, train_qmix_marl2, test_model, run_game_ai
        from training import DQN_test_render, DRQN_test_render, test_qmix_marl2
        entries = [(DQN_train, []), (DRQN_train, []), (train_qmix_marl2, []), (run_game_ai, []),
                   (test_model, ["--checkpoint", "model.pt"]),
                   (DQN_test_render, ["--checkpoint", "model.pt"]),
                   (DRQN_test_render, ["--checkpoint", "model.pt"]),
                   (test_qmix_marl2, ["--checkpoint", "model.pt"])]
        with patch.object(devices, "_load_npu", side_effect=AssertionError("CLI probed NPU")):
            for module, args in entries:
                for name in ("npu", "npu:2", "cuda:1", "mps", "cpu"):
                    with self.subTest(entry=module.__name__, device=name):
                        self.assertEqual(module.parse_args(args + ["--device", name]).device, name)

    def test_invalid_names_are_rejected_before_hardware_detection(self):
        parser = argparse.ArgumentParser()
        devices.add_device_argument(parser)
        for name in ("npu:-1", "npu:", "npu:abc", "cuda:-1", "mps:1", "unknown"):
            with self.subTest(device=name), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--device", name])

    def test_npu_is_registered_before_constructing_its_device(self):
        backend, _ = fake_npu()
        events = []
        def register(name):
            self.assertEqual(name, "torch_npu")
            events.append("register")
            setattr(torch, "npu", backend)
        def construct(name):
            events.append("device")
            return device_factory(name)
        with patch.object(torch, "npu", None, create=True), \
                patch.object(devices.importlib, "import_module", side_effect=register), \
                patch.object(torch, "device", side_effect=construct):
            result = devices.get_device("npu:2")
        self.assertEqual((result.type, result.index), ("npu", 2))
        self.assertEqual(events, ["register", "device"])
        backend.set_device.assert_called_once_with(2)

    def test_npu_without_index_selects_first_visible_device(self):
        backend, _ = fake_npu()
        with patch.object(torch, "npu", backend, create=True), \
                patch.object(torch, "device", side_effect=device_factory):
            result = devices.get_device("npu")
        self.assertEqual(result.index, 0)
        backend.set_device.assert_called_once_with(0)

    def test_out_of_range_indices_do_not_change_active_device(self):
        backend, _ = fake_npu()
        with patch.object(torch, "npu", backend, create=True):
            with self.assertRaisesRegex(ValueError, "可见设备数"):
                devices.get_device("npu:4")
        backend.set_device.assert_not_called()
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "device_count", return_value=2), \
                patch.object(torch.cuda, "set_device") as select:
            with self.assertRaisesRegex(ValueError, "CUDA"):
                devices.get_device("cuda:2")
            select.assert_not_called()

    def test_indexed_cuda_selects_requested_device(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "device_count", return_value=4), \
                patch.object(torch.cuda, "set_device") as select:
            self.assertEqual(devices.get_device("cuda:2"), TORCH_DEVICE("cuda:2"))
            select.assert_called_once_with(2)

    def test_auto_uses_cuda_then_npu(self):
        backend, _ = fake_npu()
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(devices, "_load_npu", side_effect=AssertionError("unexpected probe")):
            self.assertEqual(devices.get_device("auto"), TORCH_DEVICE("cuda"))
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch, "npu", backend, create=True), \
                patch.object(torch, "device", side_effect=device_factory):
            self.assertEqual(devices.get_device("auto").type, "npu")

    def test_missing_optional_npu_preserves_mps_and_cpu_fallback(self):
        missing = ModuleNotFoundError("No module named torch_npu", name="torch_npu")
        with patch.object(torch, "npu", None, create=True), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=True), \
                patch.object(devices.importlib, "import_module", side_effect=missing):
            self.assertEqual(devices.get_device("auto"), TORCH_DEVICE("mps"))
            with patch.object(torch.backends.mps, "is_available", return_value=False):
                self.assertEqual(devices.get_device("auto"), TORCH_DEVICE("cpu"))
            with self.assertRaisesRegex(RuntimeError, "torch_npu"):
                devices.get_device("npu")

    def test_installed_but_broken_npu_reports_runtime_configuration_error(self):
        with patch.object(torch, "npu", None, create=True), \
                patch.object(devices.importlib, "import_module", side_effect=ImportError("libascendcl.so")):
            with self.assertRaisesRegex(RuntimeError, "CANN"):
                devices.get_device("npu")

    def test_cpu_seeding_and_selection_do_not_probe_or_seed_an_accelerator(self):
        backend, _ = fake_npu()
        with patch.object(torch, "npu", backend, create=True), \
                patch.object(devices, "_load_npu", side_effect=AssertionError("CPU probed NPU")):
            self.assertEqual(devices.get_device("cpu"), TORCH_DEVICE("cpu"))
            devices.seed_torch(42)
            first = torch.rand(4)
            devices.seed_torch(42)
            torch.testing.assert_close(first, torch.rand(4))
        backend.manual_seed_all.assert_not_called()

    def test_npu_seed_and_rng_restore_cover_cpu_and_accelerator(self):
        backend, generator = fake_npu()
        device = fake_device(1)
        with patch.object(torch, "npu", backend, create=True):
            devices.seed_torch(19, device)
            state = devices.capture_torch_rng_state(device)
            cpu_expected = torch.rand(5)
            npu_expected = torch.rand(5, generator=generator)
            devices.seed_torch(100, device)
            devices.restore_torch_rng_state(state)
            torch.testing.assert_close(cpu_expected, torch.rand(5))
            torch.testing.assert_close(npu_expected, torch.rand(5, generator=generator))
        self.assertEqual(backend.manual_seed_all.call_args_list, [call(19), call(100)])
        backend.set_rng_state_all.assert_called_once()

    def test_mps_rng_and_synchronization_use_its_single_generator_api(self):
        backend = Mock()
        backend.get_rng_state.return_value = torch.random.get_rng_state()
        with patch.object(torch, "mps", backend):
            device = TORCH_DEVICE("mps")
            devices.seed_torch(41, device)
            devices.restore_torch_rng_state(devices.capture_torch_rng_state(device))
            devices.synchronize_device(device)
        backend.manual_seed.assert_called_once_with(41)
        backend.set_rng_state.assert_called_once()
        backend.synchronize.assert_called_once_with()

    def test_npu_sync_and_device_info_use_selected_backend(self):
        backend, _ = fake_npu()
        device = fake_device(2)
        with patch.object(torch, "npu", backend, create=True), redirect_stdout(io.StringIO()) as output:
            devices.synchronize_device(device)
            devices.print_device_info(device)
        backend.synchronize.assert_called_once_with(device)
        backend.get_device_name.assert_called_once_with(device)
        self.assertIn("Mock Ascend", output.getvalue())

    def test_evaluation_restores_npu_and_cpu_rng_even_when_runner_fails(self):
        from training.train_qmix_marl2 import evaluate
        backend, generator = fake_npu()
        before_cpu = torch.random.get_rng_state()
        before_npu = generator.get_state()
        before_py = random.getstate()
        before_np = np.random.get_state()
        runner = Mock()
        def fail(**kwargs):
            torch.rand(5); torch.rand(5, generator=generator)
            random.random(); np.random.random()
            raise RuntimeError("evaluation failed")
        runner.run.side_effect = fail
        with patch.object(torch, "npu", backend, create=True):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                evaluate(runner, 2, seed_base=50, device=fake_device())
        torch.testing.assert_close(before_cpu, torch.random.get_rng_state())
        torch.testing.assert_close(before_npu, generator.get_state())
        self.assertEqual(before_py, random.getstate())
        self.assertEqual(before_np[0], np.random.get_state()[0])
        np.testing.assert_array_equal(before_np[1], np.random.get_state()[1])
        self.assertEqual(before_np[2:], np.random.get_state()[2:])
        backend.manual_seed_all.assert_called_once_with(50)
        backend.set_rng_state_all.assert_called_once()

    def test_parallel_evaluation_also_seeds_and_restores_npu_rng(self):
        from training.train_qmix_marl2 import evaluate
        from training.marl2.runners.parallel_episode_runner import ParallelEpisodeRunner
        backend, generator = fake_npu()
        before = generator.get_state()
        class Runner(ParallelEpisodeRunner):
            def __init__(self):
                self.n_envs = 2
            def run(self, **kwargs):
                torch.rand(5, generator=generator)
                return None, [{"episode_return": 1., "episode_length": 2, "battle_won": True}
                              for _ in kwargs["episode_seeds"]]
        with patch.object(torch, "npu", backend, create=True):
            report = evaluate(Runner(), 3, seed_base=70, device=fake_device())
        self.assertEqual(report["battle_won_mean"], 1)
        backend.manual_seed_all.assert_called_once_with(70)
        torch.testing.assert_close(before, generator.get_state())

    def test_unified_evaluator_registers_backend_before_loading_weights(self):
        from training.test_model import evaluate, parse_args
        events = []
        def register(name):
            self.assertEqual(name, "npu:1")
            events.append("register")
            return TORCH_DEVICE("cpu")
        def load(path):
            events.append("load")
            raise ValueError("test checkpoint")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()
            args = parse_args(["--checkpoint", str(checkpoint), "--device", "npu:1"])
            with patch.object(devices, "get_device", side_effect=register), \
                    patch("training.model_evaluation.load_checkpoint", side_effect=load):
                with self.assertRaisesRegex(ValueError, "test checkpoint"):
                    evaluate(args)
        self.assertEqual(events, ["register", "load"])

    def test_portable_checkpoint_converts_nested_tensors_and_preserves_metadata(self):
        weights = OrderedDict(weight=SimulatedAcceleratorTensor([[1., 2.]]))
        weights._metadata = OrderedDict({"": {"version": 1}})
        optimizer = {"state": {0: {"square_avg": SimulatedAcceleratorTensor([3., 4.])}},
                     "param_groups": [{"params": [0], "lr": .01}]}
        payload = {"agent": weights, "optimizer": optimizer, "meta": {"step": 5}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(payload, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(restored["agent"]._metadata, weights._metadata)
        self.assertEqual(restored["optimizer"]["param_groups"], optimizer["param_groups"])
        self.assertEqual(restored["meta"], payload["meta"])
        tensor = restored["agent"]["weight"]
        self.assertIs(type(tensor), torch.Tensor)
        self.assertEqual(tensor.device.type, "cpu")
        torch.testing.assert_close(tensor, torch.tensor([[1., 2.]]))
        torch.testing.assert_close(restored["optimizer"]["state"][0]["square_avg"], torch.tensor([3., 4.]))
        self.assertEqual(weights["weight"].device.type, "npu")


if __name__ == "__main__":
    unittest.main()
