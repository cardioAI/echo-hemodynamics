#!/usr/bin/env python3
"""Echo-Hemodynamics pipeline orchestrator.

Drives the five stages — train, ablation, attention visualization, validation, test —
by invoking each runner as a Python module so they share the same package import.

Results land in ``E:\\results_cardioAI\\EchoCath_cardioAI\\{timestamp}\\`` with one
subdirectory per stage.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


RUNNERS = {
    "training": "echo_hemodynamics.runners.train",
    "ablation": "echo_hemodynamics.runners.ablation",
    "attention_viz": "echo_hemodynamics.runners.visualize",
    "validation": "echo_hemodynamics.runners.validate",
    "test": "echo_hemodynamics.runners.test",
}

STAGE_DIR_NAMES = {
    "training": "train_cardioAI",
    "ablation": "attention_ablation_cardioAI",
    "attention_viz": "attention_visualizations_cardioAI",
    "validation": "validation_cardioAI",
    "test": "test_cardioAI",
}


class CardioAIPipeline:
    """Coordinates the five pipeline stages and writes a per-run summary."""

    def __init__(self, base_output_dir=None):
        if base_output_dir is None:
            base_output_dir = r"E:\results_cardioAI\EchoCath_cardioAI"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(base_output_dir)
        self.run_dir = self.base_output_dir / self.timestamp

        self.epochs = 100
        self.stage_epochs = 20
        self.batch_size = 32
        self.training_frames = 32
        self.train_size = 235
        self.train_indices = None
        self.num_folds = 5

        self.skip_train = False
        self.skip_ablation = False
        self.skip_visualizations = False
        self.training_only = False
        self.run_validation = True
        self.run_test = True
        self.attention_frames = 8
        self.ablation_patients = 235
        self.ablation_batch_size = 2
        self.stages = 12
        self.ablation_attentions = "temporal,fusion"

        self.dirs = {key: self.run_dir / name for key, name in STAGE_DIR_NAMES.items()}
        self.dirs["logs"] = self.run_dir / "logs"

        self.results = {}
        self.pipeline_log = []

    def create_directories(self):
        print(f"Creating main timestamp directory: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created main directory: {self.run_dir.name}")

    def log_step(self, step, status, details="", duration=None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "details": details,
            "duration_seconds": duration,
        }
        self.pipeline_log.append(log_entry)

        status_symbol = "OK" if status == "SUCCESS" else "ERROR" if status == "ERROR" else "RUNNING"
        duration_str = f" ({duration:.1f}s)" if duration else ""
        print(f"{status_symbol} {step}: {status}{duration_str}")
        if details and status == "ERROR":
            print(f"    {details}")

    def run_stage(self, stage_key, extra_env=None):
        """Run a single stage as ``python -m <module>`` and capture logs."""
        module = RUNNERS[stage_key]
        subfolder = self.run_dir / STAGE_DIR_NAMES[stage_key]
        subfolder.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CARDIOAI_OUTPUT_DIR"] = str(subfolder)
        env["CARDIOAI_TIMESTAMP"] = self.timestamp
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})

        print(f"\n{'=' * 60}")
        print(f"EXECUTING: {module}")
        print(f"Output Directory: {subfolder}")
        print(f"{'=' * 60}")

        start = datetime.now()
        result = subprocess.run(
            [sys.executable, "-m", module],
            env=env, capture_output=True, text=True,
        )
        duration = (datetime.now() - start).total_seconds()

        log_file = subfolder / f"{STAGE_DIR_NAMES[stage_key]}_execution.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"MODULE: {module}\n")
            f.write(f"START: {start.isoformat()}\n")
            f.write(f"DURATION: {duration:.2f}s\n")
            f.write(f"RETURN CODE: {result.returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)

        if result.returncode == 0:
            self.log_step(stage_key, "SUCCESS", duration=duration)
            return True, result.stdout, result.stderr
        error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
        self.log_step(stage_key, "ERROR", error_msg, duration)
        return False, result.stdout, result.stderr

    def find_and_copy_latest_weights(self):
        print("Finding latest training weights...")
        if not self.base_output_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.base_output_dir}")

        timestamp_dirs = [
            d for d in self.base_output_dir.iterdir() if d.is_dir() and d.name != self.timestamp
        ]
        if not timestamp_dirs:
            raise FileNotFoundError("No existing training results found")

        latest_dir = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
        latest_training_dir = latest_dir / "train_cardioAI"
        if not latest_training_dir.exists():
            raise FileNotFoundError(f"No training directory found in {latest_dir}")

        latest_model_path = latest_training_dir / "best_model.pth"
        if not latest_model_path.exists():
            raise FileNotFoundError(f"No best_model.pth found in {latest_training_dir}")

        current_training_dir = self.run_dir / "train_cardioAI"
        current_training_dir.mkdir(parents=True, exist_ok=True)

        target_model_path = current_training_dir / "best_model.pth"
        shutil.copy2(latest_model_path, target_model_path)
        print(f"Copied model weights from: {latest_model_path}")

        for history_file in ["training_history.json", "training_results.json"]:
            src = latest_training_dir / history_file
            if src.exists():
                shutil.copy2(src, current_training_dir / history_file)
                print(f"Copied: {history_file}")

        return latest_dir.name

    def save_pipeline_summary(self):
        summary = {
            "pipeline_info": {
                "timestamp": self.timestamp,
                "run_directory": str(self.run_dir),
                "execution_time": datetime.now().isoformat(),
            },
            "directory_structure": {k: str(v) for k, v in self.dirs.items()},
            "execution_log": self.pipeline_log,
            "results_summary": self.results,
        }

        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        readable_file = logs_dir / "execution_summary.txt"
        with open(readable_file, "w") as f:
            f.write("ECHO-HEMODYNAMICS PIPELINE EXECUTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Run ID: {self.timestamp}\n")
            f.write(f"Output Directory: {self.run_dir}\n")
            f.write(f"Execution Time: {datetime.now().isoformat()}\n\n")
            for entry in self.pipeline_log:
                status_symbol = "OK" if entry["status"] == "SUCCESS" else "ERROR" if entry["status"] == "ERROR" else "RUNNING"
                duration = f" ({entry['duration_seconds']:.1f}s)" if entry.get("duration_seconds") else ""
                f.write(f"{status_symbol} {entry['step']}: {entry['status']}{duration}\n")
                if entry.get("details") and entry["status"] == "ERROR":
                    f.write(f"    Error: {entry['details']}\n")

    def run_complete_pipeline(self):
        print("ECHO-HEMODYNAMICS COMPLETE PIPELINE EXECUTION")
        print("=" * 60)
        print(f"Run ID: {self.timestamp}")
        print(f"Output Directory: {self.run_dir}")
        print("=" * 60)

        try:
            try:
                import torch  # local import keeps argparse / dry-pass torch-free
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    print(f"GPU Available: {torch.cuda.get_device_name()}")
                    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            except ImportError:
                print("Note: PyTorch not available in this environment; GPU setup skipped.")

            self.create_directories()

            # Step 1: training (or load weights)
            if self.skip_train:
                print(f"\n{'=' * 50}")
                print("STEP 1: LOADING LATEST WEIGHTS (SKIPPING TRAINING)")
                print(f"{'=' * 50}")
                try:
                    latest_run = self.find_and_copy_latest_weights()
                    self.log_step("load_latest_weights", "SUCCESS", f"Loaded from {latest_run}")
                except Exception as e:
                    self.log_step("load_latest_weights", "ERROR", str(e))
                    return False
            else:
                print(f"\n{'=' * 50}")
                print("STEP 1: TRAINING")
                print(f"{'=' * 50}")
                training_env = {
                    "CARDIOAI_EPOCHS": self.epochs,
                    "CARDIOAI_STAGE_EPOCHS": self.stage_epochs,
                    "CARDIOAI_BATCH_SIZE": self.batch_size,
                    "CARDIOAI_TRAINING_FRAMES": self.training_frames,
                    "CARDIOAI_TRAIN_SIZE": self.train_size,
                    "CARDIOAI_STAGES": self.stages,
                    "CARDIOAI_ABLATION_ATTENTIONS": self.ablation_attentions,
                    "CARDIOAI_NUM_FOLDS": self.num_folds,
                    "CARDIOAI_TRAIN_INDICES": (
                        self.train_indices if self.train_indices else f"0-{self.train_size - 1}"
                    ),
                }
                success, _, stderr = self.run_stage("training", extra_env=training_env)
                if not success:
                    print(f"ERROR: Training failed.\nLast error: {stderr[-500:] if stderr else 'None'}")
                    return False

            if self.training_only:
                print(f"\n{'=' * 50}")
                print("TRAINING-ONLY MODE: Skipping all subsequent steps")
                print(f"{'=' * 50}")
                self.save_pipeline_summary()
                return True

            # Step 2: ablation
            if not self.skip_ablation:
                print(f"\n{'=' * 50}")
                print("STEP 2: ATTENTION ABLATION STUDY")
                print(f"{'=' * 50}")
                # Ablation runs the same epoch count as main training so the
                # comparison between full model and ablation variants is fair.
                ablation_env = {
                    "CARDIOAI_ABLATION_EPOCHS": self.epochs,
                    "CARDIOAI_ABLATION_PATIENTS": self.ablation_patients,
                    "CARDIOAI_ABLATION_BATCH_SIZE": self.ablation_batch_size,
                    "CARDIOAI_TRAINING_FRAMES": self.training_frames,
                    "CARDIOAI_NUM_FOLDS": self.num_folds,
                    "CARDIOAI_TRAIN_INDICES": (
                        self.train_indices if self.train_indices else f"0-{self.train_size - 1}"
                    ),
                }
                success, _, stderr = self.run_stage("ablation", extra_env=ablation_env)
                if not success:
                    print(f"WARNING: Ablation study failed: {stderr[-200:] if stderr else 'None'}")
            else:
                self.log_step("ablation", "SKIPPED", "User requested skip")

            # Step 3: attention visualizations
            if not self.skip_visualizations:
                print(f"\n{'=' * 50}")
                print("STEP 3: ATTENTION VISUALIZATIONS")
                print(f"{'=' * 50}")
                viz_env = {
                    "CARDIOAI_ATTENTION_FRAMES": self.attention_frames,
                    "CARDIOAI_TRAINING_FRAMES": self.training_frames,
                }
                success, _, stderr = self.run_stage("attention_viz", extra_env=viz_env)
                if not success:
                    print(f"ERROR: Visualizations failed: {stderr[-300:] if stderr else 'None'}")
            else:
                self.log_step("attention_viz", "SKIPPED", "User requested skip")

            # Step 4: validation
            if self.run_validation:
                print(f"\n{'=' * 50}")
                print("STEP 4: INTERNAL VALIDATION (COHORT I, n=235)")
                print(f"{'=' * 50}")
                val_env = {"CARDIOAI_TRAINING_FRAMES": self.training_frames}
                success, _, stderr = self.run_stage("validation", extra_env=val_env)
                if not success:
                    print(f"WARNING: Validation failed: {stderr[-200:] if stderr else 'None'}")

            # Step 5: independent test
            if self.run_test:
                print(f"\n{'=' * 50}")
                print("STEP 5: INDEPENDENT TEST (COHORT II, n=73)")
                print(f"{'=' * 50}")
                test_env = {"CARDIOAI_TRAINING_FRAMES": self.training_frames}
                success, _, stderr = self.run_stage("test", extra_env=test_env)
                if not success:
                    print(f"WARNING: Test failed: {stderr[-200:] if stderr else 'None'}")

            self.save_pipeline_summary()

            print(f"\n{'=' * 60}")
            print("PIPELINE EXECUTION COMPLETED")
            print(f"{'=' * 60}")
            print(f"Results saved to: {self.run_dir}")
            successful = sum(1 for entry in self.pipeline_log if entry["status"] == "SUCCESS")
            print(f"Pipeline Success Rate: {successful}/{len(self.pipeline_log)} steps")

            return True

        except Exception as e:
            print(f"FATAL ERROR in pipeline execution: {e}")
            traceback.print_exc()
            self.log_step("pipeline_execution", "FATAL_ERROR", str(e))
            return False


def build_parser():
    parser = argparse.ArgumentParser(description="Echo-Hemodynamics Complete Pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage-epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=235)
    parser.add_argument("--train-indices", type=str, default=None)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--training-frames", type=int, default=32)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--ablation-patients", type=int, default=235)
    parser.add_argument("--ablation-batch-size", type=int, default=16)
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--ablation_attentions", type=str, default="temporal,fusion")
    parser.add_argument("--run-validation", dest="run_validation", action="store_true", default=True)
    parser.add_argument("--no-validation", dest="run_validation", action="store_false")
    parser.add_argument("--run-test", dest="run_test", action="store_true", default=True)
    parser.add_argument("--no-test", dest="run_test", action="store_false")
    return parser


def main():
    try:
        args = build_parser().parse_args()

        pipeline = CardioAIPipeline()
        pipeline.epochs = args.epochs
        pipeline.stage_epochs = args.stage_epochs
        pipeline.batch_size = args.batch_size
        pipeline.training_frames = args.training_frames
        pipeline.train_size = args.train_size
        pipeline.train_indices = args.train_indices
        pipeline.num_folds = args.num_folds
        pipeline.skip_train = args.skip_train
        pipeline.skip_ablation = args.skip_ablation
        pipeline.skip_visualizations = args.skip_visualizations
        pipeline.training_only = args.training_only
        pipeline.attention_frames = args.frames
        pipeline.ablation_patients = args.ablation_patients
        pipeline.ablation_batch_size = args.ablation_batch_size
        pipeline.stages = args.stages
        pipeline.ablation_attentions = args.ablation_attentions
        pipeline.run_validation = args.run_validation
        pipeline.run_test = args.run_test

        success = pipeline.run_complete_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
