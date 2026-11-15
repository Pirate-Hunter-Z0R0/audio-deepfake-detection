"""
voice_generation/applio_wrapper.py
----------------------------------
Python wrapper for Applio CLI (core.py) enabling decoupled voice conversion
and two-stage text-to-speech generation.

Features:
    - Dynamic Model Auto-Discovery: Automatically scans models_dir for any
      trained .pth and .index pairs (teammate01, teammate02, teammate04, etc.)
      without requiring manual config edits.
    - Two-stage synthesis (text_to_rvc) with intermediate audio preservation.
    - Direct voice conversion (convert_audio) on pre-existing recordings.
"""

import os
import sys
import glob
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ApplioWrapper:
    """
    Encapsulates interaction with Applio's core.py CLI via subprocess.
    All paths and hyperparameters are loaded from config.yaml, with automatic
    dynamic discovery of any newly trained models.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not cfg_file.exists():
            raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.applio_cfg = self.config.get("applio", {})
        self.gen_cfg = self.config.get("generation", {})
        self.models_cfg = self.config.get("models", {})
        self.defaults = self.config.get("inference_defaults", {})

        self.applio_dir = Path(self.applio_cfg.get("applio_dir", "/content/Applio"))
        self.models_dir = Path(self.applio_cfg.get("models_dir", "/content/drive/MyDrive/ApplioBackup"))
        self.python_exec = self.applio_cfg.get("python_exec", sys.executable)
        self.timeout = int(self.applio_cfg.get("timeout_seconds", 120))

        self.intermediate_tts_dir = Path(self.gen_cfg.get("intermediate_tts_dir", "data/intermediate_tts"))
        self.output_dir = Path(self.gen_cfg.get("output_dir", "data/synthetic_rvc"))

        self.intermediate_tts_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Run dynamic model discovery to automatically find any new voices
        self._auto_discover_models()

    def _auto_discover_models(self) -> None:
        """
        Dynamically scans models_dir and registers any trained voice folders
        containing .pth weights and .index files into self.models_cfg.
        Supports:
          - Subfolders (e.g. models_dir/teammate04_rvc/teammate04_rvc_100e.pth)
          - Direct files (e.g. models_dir/teammate04.pth)
          - Filtering out Applio training artifacts (G_*.pth, D_*.pth)
        """
        if not self.models_dir.exists():
            return

        # 1. Scan subdirectories in models_dir (standard Applio export structure)
        for folder in self.models_dir.iterdir():
            if folder.is_dir():
                spk_name = folder.name.replace("_rvc", "")
                
                # Exclude G_ and D_ checkpoints
                candidates = [
                    p for p in folder.glob("*.pth")
                    if not (p.name.startswith("G_") or p.name.startswith("D_"))
                ]
                # Sort: prefer 100e or larger epoch count, or latest file
                candidates.sort(key=lambda x: ("100e" in x.name, x.stat().st_mtime), reverse=True)
                index_files = sorted(list(folder.glob("*.index")))

                if candidates:
                    best_pth = candidates[0]
                    best_idx = index_files[0] if index_files else None

                    if spk_name not in self.models_cfg:
                        self.models_cfg[spk_name] = {
                            "pth_file": str(best_pth.relative_to(self.models_dir)),
                            "index_file": str(best_idx.relative_to(self.models_dir)) if best_idx else "",
                            "gender": "male",
                            "default_tts_voice": "en-IN-PrabhatNeural"
                        }

        # 2. Also check if loose .pth files are placed directly in models_dir
        for pth in self.models_dir.glob("*.pth"):
            if pth.name.startswith("G_") or pth.name.startswith("D_"):
                continue
            spk_name = pth.stem.replace("_rvc", "")
            if spk_name not in self.models_cfg:
                matching_indices = list(self.models_dir.glob(f"*{spk_name}*.index"))
                best_idx = matching_indices[0] if matching_indices else None
                self.models_cfg[spk_name] = {
                    "pth_file": pth.name,
                    "index_file": best_idx.name if best_idx else "",
                    "gender": "male",
                    "default_tts_voice": "en-IN-PrabhatNeural"
                }

    def get_available_models(self) -> List[str]:
        """Returns list of all registered and auto-discovered model names."""
        self._auto_discover_models()
        return sorted(list(self.models_cfg.keys()))

    def validate_model(self, model_name: str) -> Tuple[Path, Optional[Path]]:
        """
        Validates that the specified model exists and that its
        .pth checkpoint and .index retrieval file physically exist on disk.
        """
        self._auto_discover_models()

        if model_name not in self.models_cfg:
            available = self.get_available_models()
            raise KeyError(
                f"Model '{model_name}' not found. Available models in {self.models_dir}: {available}"
            )

        m_info = self.models_cfg[model_name]
        pth_path = (self.models_dir / m_info["pth_file"]).resolve()
        idx_val = m_info.get("index_file", "")
        index_path = (self.models_dir / idx_val).resolve() if idx_val else None

        if not pth_path.is_file():
            raise FileNotFoundError(f"RVC model weights (.pth) not found: {pth_path}")
        if index_path and not index_path.is_file():
            print(f"[WARNING] FAISS index not found at: {index_path}. Proceeding without index.")
            index_path = None

        return pth_path, index_path

    def _run_cli_command(self, cmd: list, task_name: str) -> None:
        core_py = self.applio_dir / "core.py"
        if not core_py.is_file():
            raise FileNotFoundError(f"Applio core.py not found at: {core_py}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.applio_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Applio CLI [{task_name}] timed out after {self.timeout}s.") from e

        if result.returncode != 0:
            err_msg = (
                f"Applio CLI [{task_name}] failed with return code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}\n"
            )
            raise RuntimeError(err_msg)

    def text_to_rvc(
        self,
        text: str,
        model_name: str,
        out_path: str,
        tts_voice: Optional[str] = None,
        intermediate_tts_path: Optional[str] = None,
        **overrides: Any
    ) -> Dict[str, str]:
        pth_path, index_path = self.validate_model(model_name)
        out_file = Path(out_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if intermediate_tts_path:
            tts_file = Path(intermediate_tts_path).resolve()
        else:
            tts_file = (self.intermediate_tts_dir / f"tts_{out_file.stem}.wav").resolve()
        tts_file.parent.mkdir(parents=True, exist_ok=True)

        prompt_txt = tts_file.with_suffix(".txt")
        prompt_txt.write_text(str(text), encoding="utf-8")

        voice = tts_voice or self.models_cfg[model_name].get("default_tts_voice", "en-IN-PrabhatNeural")
        pitch = overrides.get("pitch", self.defaults.get("pitch", 0))
        index_rate = overrides.get("index_rate", self.defaults.get("index_rate", 0.80))
        protect = overrides.get("protect", self.defaults.get("protect", 0.33))
        f0_method = overrides.get("f0_method", self.defaults.get("f0_method", "rmvpe"))
        embedder = overrides.get("embedder_model", self.defaults.get("embedder_model", "contentvec"))
        export_format = overrides.get("export_format", self.defaults.get("export_format", "WAV"))
        tts_rate = overrides.get("tts_rate", self.defaults.get("tts_rate", 0))

        cmd = [
            self.python_exec, "core.py", "tts",
            "--tts-file", str(prompt_txt),
            "--tts-text", str(text),
            "--tts-voice", str(voice),
            "--tts-rate", str(int(tts_rate)),
            "--output-tts-path", str(tts_file),
            "--output-rvc-path", str(out_file),
            "--pth-path", str(pth_path),
            "--index-path", str(index_path) if index_path else "",
            "--pitch", str(int(pitch)),
            "--index-rate", str(float(index_rate)) if index_path else "0",
            "--protect", str(float(protect)),
            "--f0-method", str(f0_method),
            "--embedder-model", str(embedder),
            "--export-format", str(export_format)
        ]

        self._run_cli_command(cmd, task_name=f"text_to_rvc ({model_name})")

        if not out_file.is_file() or out_file.stat().st_size == 0:
            raise RuntimeError(f"RVC output was not generated at: {out_file}")

        return {
            "output_rvc_path": str(out_file),
            "intermediate_tts_path": str(tts_file)
        }

    def convert_audio(
        self,
        input_wav: str,
        model_name: str,
        out_path: str,
        **overrides: Any
    ) -> str:
        in_file = Path(input_wav).resolve()
        if not in_file.is_file():
            raise FileNotFoundError(f"Source input audio not found at: {in_file}")

        pth_path, index_path = self.validate_model(model_name)
        out_file = Path(out_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        pitch = overrides.get("pitch", self.defaults.get("pitch", 0))
        index_rate = overrides.get("index_rate", self.defaults.get("index_rate", 0.80))
        protect = overrides.get("protect", self.defaults.get("protect", 0.33))
        f0_method = overrides.get("f0_method", self.defaults.get("f0_method", "rmvpe"))
        embedder = overrides.get("embedder_model", self.defaults.get("embedder_model", "contentvec"))
        export_format = overrides.get("export_format", self.defaults.get("export_format", "WAV"))

        cmd = [
            self.python_exec, "core.py", "infer",
            "--input-path", str(in_file),
            "--output-path", str(out_file),
            "--pth-path", str(pth_path),
            "--index-path", str(index_path) if index_path else "",
            "--pitch", str(int(pitch)),
            "--index-rate", str(float(index_rate)) if index_path else "0",
            "--protect", str(float(protect)),
            "--f0-method", str(f0_method),
            "--embedder-model", str(embedder),
            "--export-format", str(export_format)
        ]

        self._run_cli_command(cmd, task_name=f"convert_audio ({model_name})")

        if not out_file.is_file() or out_file.stat().st_size == 0:
            raise RuntimeError(f"RVC output was not generated at: {out_file}")

        return str(out_file)
