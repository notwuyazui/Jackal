"""Dependency diagnostics without requiring an Ascend runtime or old setuptools."""
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

from training.utils import runtime_warnings as runtime

ROOT = Path(__file__).resolve().parents[2]
CANN = "/usr/local/Ascend/ascend-toolkit/latest"
RESOLVED_CANN = "/usr/local/Ascend/ascend-toolkit/8.2.RC1"
INSTALLATION = (CANN, RESOLVED_CANN, 0, 1001)


def owner_warning(path=CANN, module="torch_npu.utils.collect_env"):
    warnings.warn_explicit(
        f"Warning: The {path} owner does not match the current owner.",
        UserWarning, filename="/site-packages/torch_npu/utils/collect_env.py", lineno=58,
        module=module,
    )


class DependencyWarningTests(unittest.TestCase):
    def setUp(self):
        self.raw = patch.dict(os.environ, {"JACKAL_RAW_DEPENDENCY_WARNINGS": "0"})
        self.raw.start()
        self.addCleanup(self.raw.stop)

    def test_only_known_ownership_notices_for_the_readable_sdk_are_replaced(self):
        with patch.object(runtime, "_shared_cann_installation", return_value=INSTALLATION), \
                warnings.catch_warnings(record=True) as seen:
            warnings.resetwarnings()
            runtime.configure_runtime_warnings()
            for _ in range(2):
                owner_warning()
                owner_warning(RESOLVED_CANN)
                owner_warning(RESOLVED_CANN + "/aarch64-linux/ascend_toolkit_install.info")
            self.assertEqual(seen, [])
            owner_warning("/another/install")
            owner_warning(module="torch_npu.utils.path_manager")
            warnings.warn_explicit("operator fallback", UserWarning,
                                   filename="collect_env.py", lineno=59,
                                   module="torch_npu.utils.collect_env")
            self.assertEqual(len(seen), 3)
            self.assertIn("operator fallback", str(seen[-1].message))

    def test_unverified_paths_keep_the_original_warning(self):
        with patch.object(runtime, "_shared_cann_installation", return_value=None), \
                warnings.catch_warnings(record=True) as seen:
            warnings.resetwarnings()
            runtime.configure_runtime_warnings()
            owner_warning()
            self.assertEqual(len(seen), 1)

    def test_explicit_warning_error_policy_is_not_overridden(self):
        with patch.object(runtime, "_shared_cann_installation", return_value=INSTALLATION), \
                warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            runtime.configure_runtime_warnings()
            with self.assertRaisesRegex(UserWarning, "owner does not match"):
                owner_warning()

    def test_raw_diagnostics_bypass_compatibility_filters_and_summary(self):
        with patch.dict(os.environ, {"JACKAL_RAW_DEPENDENCY_WARNINGS": "1"}), \
                patch.object(runtime, "_shared_cann_installation", return_value=INSTALLATION), \
                warnings.catch_warnings(record=True) as seen, redirect_stderr(io.StringIO()) as output:
            warnings.resetwarnings()
            runtime.configure_runtime_warnings()
            owner_warning()
            runtime.report_npu_installation()
            self.assertEqual(len(seen), 1)
            self.assertEqual(output.getvalue(), "")

    def test_shared_installation_summary_is_printed_once_per_resolved_sdk(self):
        with patch.object(runtime, "_shared_cann_installation", return_value=INSTALLATION), \
                patch.object(runtime, "_reported_installations", set()), \
                redirect_stderr(io.StringIO()) as output:
            runtime.report_npu_installation()
            runtime.report_npu_installation()
        self.assertEqual(output.getvalue().count("[NPU]"), 1)
        self.assertIn("UID=0", output.getvalue())
        self.assertIn("UID=1001", output.getvalue())
        self.assertIn("权限及运行时检查仍保留", output.getvalue())

    @unittest.skipUnless(hasattr(os, "getuid"), "POSIX ownership checks")
    def test_shared_sdk_detection_rejects_unreadable_world_writable_and_missing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o755)
            owner = root.stat().st_uid
            with patch.dict(os.environ, {"ASCEND_HOME_PATH": directory}), \
                    patch.object(runtime.os, "getuid", return_value=owner + 1):
                installation = runtime._shared_cann_installation()
                self.assertIsNotNone(installation)
                assert installation is not None
                self.assertEqual(installation[2:], (owner, owner + 1))
                with patch.object(runtime.os, "access", return_value=False):
                    self.assertIsNone(runtime._shared_cann_installation())
                root.chmod(0o777)
                self.assertIsNone(runtime._shared_cann_installation())
                root.chmod(0o755)
            with patch.dict(os.environ, {"ASCEND_HOME_PATH": directory + "/missing"}):
                self.assertIsNone(runtime._shared_cann_installation())

    @unittest.skipUnless(hasattr(os, "getuid"), "POSIX ownership checks")
    def test_direct_training_entries_configure_diagnostics_before_torch_autoload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cann = root / "cann"
            cann.mkdir(mode=0o755)
            owner = cann.stat().st_uid
            Path(directory, "torch.py").write_text(
                'import os, warnings\n'
                'for _ in range(2):\n'
                '    warnings.warn_explicit("Warning: The " + os.environ["ASCEND_HOME_PATH"] + '
                '" owner does not match the current owner.", UserWarning, '
                'filename="/site-packages/torch_npu/utils/collect_env.py", lineno=58, '
                'module="torch_npu.utils.collect_env")\n'
                'raise RuntimeError("simulated SDK permission check failed")\n', encoding="utf-8")
            for entry in ("train_qmix_marl2.py", "training/train_qmix_marl2.py",
                          "training/DQN_train.py", "training/DRQN_train.py"):
                with self.subTest(entry=entry):
                    script = (f"import os, runpy, sys\nos.getuid = lambda: {owner + 1}\n"
                              f"sys.argv = [{entry!r}, '--help']\n"
                              f"runpy.run_path({str(ROOT / entry)!r}, run_name='__main__')\n")
                    env = dict(os.environ, ASCEND_HOME_PATH=str(cann),
                               PYTHONPATH=os.pathsep.join((directory, str(ROOT))))
                    env.pop("PYTHONWARNINGS", None)
                    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT,
                                            capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("RuntimeError: simulated SDK permission check failed", result.stderr)
                    self.assertNotIn("UserWarning:", result.stderr)

    def test_geometry_import_handles_legacy_pygame_deprecation_and_preserves_font_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            # Emulate the legacy dependency on hosts where pkg_resources has
            # already been removed. Pygame's supported filesystem fallback runs.
            Path(directory, "pkg_resources.py").write_text(
                'import warnings\n'
                'warnings.warn("pkg_resources is deprecated as an API. See https://setuptools.pypa.io/", '
                'UserWarning, stacklevel=2)\n'
                'raise ImportError("emulated missing pkg_resources")\n', encoding="utf-8")
            script = (
                "import warnings\n"
                "before = list(warnings.filters)\n"
                "from engine.geometry import Rect\n"
                "assert list(warnings.filters) == before\n"
                "import pygame\n"
                "assert Rect(1.5, 2.5, 3.5, 4.5) == pygame.Rect(1.5, 2.5, 3.5, 4.5)\n"
                "assert not pygame.display.get_init()\n"
                "pygame.font.init()\n"
                "assert pygame.font.Font(None, 12).render('Jackal', True, (255,255,255)).get_width() > 0\n"
                "warnings.warn('unrelated resource warning', UserWarning)\n"
            )
            for raw in ("0", "1"):
                with self.subTest(raw=raw):
                    env = dict(os.environ, JACKAL_RAW_DEPENDENCY_WARNINGS=raw,
                               PYTHONPATH=os.pathsep.join((directory, str(ROOT))))
                    env.pop("PYTHONWARNINGS", None)
                    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT,
                                            capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual("pkg_resources is deprecated" in result.stderr, raw == "1")
                    self.assertIn("unrelated resource warning", result.stderr)


if __name__ == "__main__":
    unittest.main()
