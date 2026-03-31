#!/usr/bin/env python3
"""
CardioAI pipeline orchestrator: training, ablation, visualization, validation, and testing.
Results saved to E:\\results_cardioAI\\EchoCath_cardioAI\\{timestamp}\\ with per-script subfolders.
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import traceback
import torch


class CardioAIPipeline:
    """Main pipeline orchestrator for CardioAI project"""
    
    def __init__(self, base_output_dir=None):
        if base_output_dir is None:
            # Use current directory for testing
            base_output_dir = r"E:\results_cardioAI\EchoCath_cardioAI"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(base_output_dir)
        self.run_dir = self.base_output_dir / self.timestamp
        self.epochs = 100  # Total epochs for full model training
        self.stage_epochs = 20  # Epochs per stage for progressive unfreezing
        self.batch_size = 32   # Optimized batch size for training efficiency
        self.training_frames = 32  # Maximum temporal frames for full cardiac cycle information
        self.train_size = 235  # Number of patients for training (1-235, patients 236-308 are independent test)
        self.train_indices = None  # Custom training indices (e.g., "0-179,180-234")
        self.num_folds = 5  # Number of folds for cross-validation

        # Pipeline control flags
        self.skip_train = False
        self.skip_ablation = False
        self.skip_visualizations = False
        self.training_only = False
        self.run_validation = True  # Run validation on patients 1-235 (5-fold CV cohort) - DEFAULT
        self.run_test = True  # Run independent test on patients 236-308 (73 patients) - DEFAULT
        self.attention_frames = 8  # Default number of attention frames for visualizations
        self.ablation_factor = 0.1  # Default factor for ablation study epochs (10% of main training)
        self.ablation_patients = 235  # Number of patients for ablation studies (patients 1-235)
        self.ablation_batch_size = 2  # Default batch size for ablation study variants
        self.stages = 12  # Unfreeze all 12 ViT layers progressively for domain adaptation
        self.ablation_attentions = 'temporal,fusion'  # Default: temporal and fusion attention modules
        
        # Create script-based directory structure (each script creates its own subfolder)
        self.dirs = {
            'training': self.run_dir / "train_cardioAI",
            'ablation': self.run_dir / "attention_ablation_cardioAI",
            'attention_viz': self.run_dir / "attention_visualizations_cardioAI",
            'validation': self.run_dir / "validation_cardioAI",
            'test': self.run_dir / "test_cardioAI",
            'logs': self.run_dir / "logs"
        }
        
        # Attention visualization subdirectories (within attention_visualizations_cardioAI folder)
        self.attention_subdirs = {
            'temporal_curves': self.dirs['attention_viz'] / "temporal_curves_attention_visualizations" / "temporal_curves",
            'attention_visualizations': self.dirs['attention_viz'] / "temporal_curves_attention_visualizations" / "attention_visualizations"
        }
        
        # All subdirectories that scripts might need
        self.all_subdirs = {
            # Main script outputs
            'training': self.dirs['training'],
            'ablation': self.dirs['ablation'],
            'attention_viz': self.dirs['attention_viz'],
            'validation': self.dirs['validation'],
            'test': self.dirs['test'],
            'logs': self.dirs['logs'],

            # Attention visualization subdirectories
            'temporal_curves': self.attention_subdirs['temporal_curves'],
            'attention_visualizations': self.attention_subdirs['attention_visualizations']
        }
        
        self.results = {}
        self.pipeline_log = []
        
    def create_directories(self):
        """Create main timestamp directory - scripts will create their own subfolders"""
        print(f"Creating main timestamp directory: {self.run_dir}")
        
        # Create only the main timestamp directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created main directory: {self.run_dir.name}")
            
        print(f"Main directory created: {self.run_dir}")
        print(f"Each script will create its own dedicated subfolder under this directory")
        
    def log_step(self, step, status, details="", duration=None):
        """Log pipeline step execution"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'status': status,
            'details': details,
            'duration_seconds': duration
        }
        self.pipeline_log.append(log_entry)
        
        status_symbol = "OK" if status == "SUCCESS" else "ERROR" if status == "ERROR" else "RUNNING"
        duration_str = f" ({duration:.1f}s)" if duration else ""
        print(f"{status_symbol} {step}: {status}{duration_str}")
        if details and status == "ERROR":
            print(f"    {details}")
    
    def run_script(self, script_name, output_dir, extra_args=None):
        """Execute a Python script with organized output management"""
        start_time = datetime.now()
        
        try:
            print(f"\n{'='*60}")
            print(f"EXECUTING: {script_name}")
            print(f"Output Directory: {output_dir}")
            if extra_args:
                print(f"Arguments: {extra_args}")
            print(f"{'='*60}")
            
            # Create script-specific subfolder under the main timestamp directory
            script_subfolder = self.run_dir / script_name.stem
            script_subfolder.mkdir(parents=True, exist_ok=True)
            print(f"Script-specific subfolder created: {script_subfolder}")
            
            # Update the CARDIOAI_OUTPUT_DIR to point to the script's subfolder
            os.environ['CARDIOAI_OUTPUT_DIR'] = str(script_subfolder)
            
            # Change to script directory for execution
            original_dir = os.getcwd()
            
            # Build command with extra arguments
            cmd = [sys.executable, script_name]
            if extra_args:
                cmd.extend(extra_args)
            
            # Execute script
            result = subprocess.run(
                cmd,
                cwd=original_dir,
                capture_output=True,
                text=True
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Save execution logs in the script-specific subfolder
            log_file = script_subfolder / f"{script_name.stem}_execution.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"SCRIPT: {script_name}\n")
                f.write(f"EXECUTION TIME: {start_time.isoformat()}\n")
                f.write(f"DURATION: {duration:.2f} seconds\n")
                f.write(f"RETURN CODE: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            
            if result.returncode == 0:
                self.log_step(script_name.stem, "SUCCESS", duration=duration)
                
                # Move relevant output files to organized directories
                self.organize_outputs(script_name.stem, script_subfolder)
                print(f"OK {script_name.stem} completed successfully")
                
                return True, result.stdout, result.stderr
            else:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                self.log_step(script_name.stem, "ERROR", error_msg, duration)
                return False, result.stdout, result.stderr
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_step(script_name.stem, "ERROR", str(e), duration)
            return False, "", str(e)
    
    def organize_outputs(self, script_name, output_dir):
        """Clean up any remaining files since scripts now save directly to their subfolders"""
        current_dir = Path.cwd()
        
        # General cleanup: move any remaining result files from current directory to script's output_dir
        result_patterns = ["*.json", "*.pth", "*.csv", "*.png", "*.txt", "*.eps", "*.tiff"]
        moved_files = 0
        
        for pattern in result_patterns:
            for file in current_dir.glob(pattern):
                # Skip system files and active configuration files
                if file.name not in ["CLAUDE.md", "All.xlsx", "Train.xlsx", "Internal.xlsx", "External.xlsx", "EchoCath.xlsx", "README.md", "requirements.txt", ".gitignore", "palette.jpeg"]:
                    if not file.name.endswith(('.py', '.md')):  # Don't move Python scripts or markdown
                        try:
                            dest_file = output_dir / file.name
                            if not dest_file.exists():  # Don't overwrite existing files
                                shutil.move(file, dest_file)
                                print(f"Moved remaining file {file.name} to {output_dir.name}")
                                moved_files += 1
                        except Exception as e:
                            print(f"Could not move {file.name}: {e}")
        
        if moved_files == 0:
            print(f"No additional files to move - {script_name} outputs are in correct location: {output_dir}")
        else:
            print(f"Moved {moved_files} remaining files to {script_name} subfolder")
    
    
    def find_and_copy_latest_weights(self):
        """Find the latest training weights and copy them to current run directory"""
        print("Finding latest training weights...")
        
        # Look for latest results in base directory
        base_dir = Path(r"E:\results_cardioAI\EchoCath_cardioAI")
        
        if not base_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {base_dir}")
        
        # Find all timestamp directories
        timestamp_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name != self.timestamp]
        
        if not timestamp_dirs:
            raise FileNotFoundError("No existing training results found")
        
        # Sort by timestamp (directory name) to get latest
        latest_dir = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
        
        # Look for train_cardioAI subfolder first (new structure)
        latest_training_dir = latest_dir / "train_cardioAI"
        if not latest_training_dir.exists():
            # Fall back to old structure
            latest_training_dir = latest_dir / "train_cardioAI"
        
        if not latest_training_dir.exists():
            raise FileNotFoundError(f"No training directory found in {latest_dir}")
        
        # Look for best_model.pth
        latest_model_path = latest_training_dir / "best_model.pth"
        if not latest_model_path.exists():
            raise FileNotFoundError(f"No best_model.pth found in {latest_training_dir}")
        
        # Copy the model and training files to current train_cardioAI subfolder
        import shutil
        
        # Create train_cardioAI subfolder for current run
        current_training_dir = self.run_dir / "train_cardioAI"
        current_training_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy best model
        target_model_path = current_training_dir / "best_model.pth"
        shutil.copy2(latest_model_path, target_model_path)
        print(f"Copied model weights from: {latest_model_path}")
        print(f"To: {target_model_path}")
        
        # Copy training history if available
        history_files = ['training_history.json', 'training_results.json']
        for history_file in history_files:
            src_file = latest_training_dir / history_file
            if src_file.exists():
                dst_file = current_training_dir / history_file
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {history_file}")
        
        print(f"Successfully loaded weights from: {latest_dir.name}")
        return latest_dir.name
    
    def verify_attention_visualizations(self):
        """Verify attention visualizations were generated for patient E100017564"""
        expected_display = self.attention_frames * 4 * 9 * 3 * 2 + 27
        print(f"Verifying {expected_display} Attention visualizations...")
        
        # Check in the attention_visualizations_cardioAI script subfolder
        attention_viz_script_dir = self.run_dir / 'attention_visualizations_cardioAI'
        temporal_curves_dir = attention_viz_script_dir / 'temporal_curves_attention_visualizations' / 'temporal_curves'
        attention_viz_dir = attention_viz_script_dir / 'temporal_curves_attention_visualizations' / 'attention_visualizations'
        patient_id = "E100017564"
        
        # Expected structure: frames x 4 views x 9 parameters x 3 formats x 2 assets = dynamic
        expected_viz_total = self.attention_frames * 4 * 9 * 3 * 2
        # Plus temporal curves: 9 parameters x 3 formats = 27 (no overview curve)
        expected_curves_total = 9 * 3
        expected_total = expected_viz_total + expected_curves_total
        
        # Count temporal curves files
        curves_files = []
        if temporal_curves_dir.exists():
            curves_files = list(temporal_curves_dir.glob("*.png")) + list(temporal_curves_dir.glob("*.eps")) + list(temporal_curves_dir.glob("*.tiff"))
        
        # Count attention visualization files
        viz_files = []
        if attention_viz_dir.exists():
            viz_files = list(attention_viz_dir.glob("*.png")) + list(attention_viz_dir.glob("*.eps")) + list(attention_viz_dir.glob("*.tiff"))
        
        actual_total = len(curves_files) + len(viz_files)
        
        print(f"Expected visualizations: {expected_total}")
        print(f"Generated visualizations: {actual_total}")
        
        verification_results = {
            'expected_total': expected_total,
            'actual_total': actual_total,
            'success': actual_total == expected_total,
            'patient_id': patient_id,
            'visualization_type': 'attention_rollout'
        }
        
        if verification_results['success']:
            print(f"OK VERIFICATION SUCCESS: All {expected_total} Attention visualizations generated correctly")
            self.log_step("attention_visualization_verification", "SUCCESS", f"{actual_total}/{expected_total} files verified")
        else:
            print(f"FAIL VERIFICATION FAILED: {actual_total}/{expected_total} files generated")
            self.log_step("attention_visualization_verification", "ERROR", f"Only {actual_total}/{expected_total} files generated")
        
        return verification_results
    
    def generate_aims_summary(self):
        """Generate comprehensive summary of all three aims completion"""
        print("Generating comprehensive aims summary...")
        
        summary = {
            'pipeline_execution': {
                'timestamp': self.timestamp,
                'run_directory': str(self.run_dir),
                'total_duration_hours': sum(entry.get('duration_seconds', 0) or 0 for entry in self.pipeline_log) / 3600,
                'successful_steps': sum(1 for entry in self.pipeline_log if entry['status'] == "SUCCESS"),
                'total_steps': len(self.pipeline_log)
            },
            'aims_completion': {
                'aim_1_attention_ablation': {
                    'completed': any('ablation' in entry['step'] and entry['status'] == 'SUCCESS' for entry in self.pipeline_log),
                    'description': 'Attention ablation study demonstrating multistage attention benefits'
                },
                'aim_2_visualizations': {
                    'completed': self.results.get('attention_verification', {}).get('success', False),
                    'visualizations_generated': self.results.get('attention_verification', {}).get('actual_total', 0),
                    'visualizations_expected': self.attention_frames * 4 * 9 * 3 * 2 + 27,  # Dynamic + 27 curves
                    'frames_used': self.attention_frames,
                    'description': f'Complete publication-ready results with {self.attention_frames * 4 * 9 * 3 * 2} attention visualizations'
                }
            },
            'execution_log': self.pipeline_log
        }
        
        # Save comprehensive summary
        summary_file = self.dirs['logs'] / "aims_completion_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Aims completion summary saved to: {summary_file}")
        return summary
    
    def _check_training_correlations(self):
        """Check training results to extract best validation correlations for each parameter"""
        try:
            # Look for final correlation results in train_cardioAI subfolder
            training_dir = self.run_dir / "train_cardioAI"
            param_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']

            # Try to find training_results.json first (contains both internal and external)
            results_files = list(training_dir.glob("**/training_results.json"))
            if results_files:
                print("Found training results from training script")
                with open(results_files[0], 'r') as f:
                    results = json.load(f)

                # Extract validation correlations
                if 'final_metrics' in results and 'best_correlations' in results['final_metrics']:
                    internal_corr_list = results['final_metrics']['best_correlations']
                    internal_correlations = {}
                    for i, param in enumerate(param_names):
                        if i < len(internal_corr_list):
                            internal_correlations[param] = abs(internal_corr_list[i])

                    print(f"\nCross-Validation Results (Patients 1-235, 5-Fold CV):")
                    print(f"  Best overall validation correlation: {max(internal_correlations.values()):.3f}")
                    print(f"  Parameters above 0.6: {sum(1 for corr in internal_correlations.values() if corr >= 0.6)}/9")
                    print("  Detailed correlations:")
                    for param in param_names:
                        if param in internal_correlations:
                            status = "[PASS]" if internal_correlations[param] >= 0.6 else "[FAIL]"
                            print(f"    {param:7s}: {internal_correlations[param]:.3f} {status}")

                # Extract test correlations
                if 'final_metrics' in results and 'best_test_correlations' in results['final_metrics']:
                    test_corr_list = results['final_metrics']['best_test_correlations']
                    test_correlations = {}
                    for i, param in enumerate(param_names):
                        if i < len(test_corr_list):
                            test_correlations[param] = abs(test_corr_list[i])

                    print(f"\nIndependent Test Results (Patients 236-308):")
                    print(f"  Best overall test correlation: {max(test_correlations.values()):.3f}")
                    print(f"  Parameters above 0.6: {sum(1 for corr in test_correlations.values() if corr >= 0.6)}/9")
                    print("  Detailed correlations:")
                    for param in param_names:
                        if param in test_correlations:
                            status = "[PASS]" if test_correlations[param] >= 0.6 else "[FAIL]"
                            print(f"    {param:7s}: {test_correlations[param]:.3f} {status}")

                return internal_correlations if 'internal_correlations' in locals() else None

            # Fallback: try final_correlations.json (only internal)
            final_corr_files = list(training_dir.glob("**/final_correlations.json"))
            if final_corr_files:
                print("Found final correlations from training script")
                with open(final_corr_files[0], 'r') as f:
                    correlations = json.load(f)

                print(f"\nInternal Validation Results:")
                print(f"  Best overall validation correlation: {max(correlations.values()):.3f}")
                print(f"  Parameters above 0.6: {sum(1 for corr in correlations.values() if corr >= 0.6)}/9")
                print("  Detailed correlations:")
                for param in param_names:
                    if param in correlations:
                        status = "[PASS]" if correlations[param] >= 0.6 else "[FAIL]"
                        print(f"    {param:7s}: {correlations[param]:.3f} {status}")

                return correlations

            # Fallback: look for training history
            history_files = list(training_dir.glob("**/training_history.json"))
            if history_files:
                print("Using training history for correlation check")
                with open(history_files[0], 'r') as f:
                    history = json.load(f)

                # Extract best per-task validation correlations
                if 'per_task_val_corr' in history and history['per_task_val_corr']:
                    # Get the final epoch correlations
                    final_correlations = history['per_task_val_corr'][-1]

                    # Create correlation dictionary
                    correlations = {}
                    for i, param in enumerate(param_names):
                        if i < len(final_correlations):
                            correlations[param] = abs(final_correlations[i])  # Use absolute value

                    print(f"\nInternal Validation Results:")
                    print(f"  Best overall validation correlation: {max(correlations.values()):.3f}")
                    print(f"  Parameters above 0.6: {sum(1 for corr in correlations.values() if corr >= 0.6)}/9")

                    return correlations

            print("Warning: No training correlation data found")
            return None

        except Exception as e:
            print(f"Warning: Could not check training correlations: {e}")
            return None

    def save_pipeline_summary(self):
        """Save comprehensive pipeline execution summary"""
        summary = {
            'pipeline_info': {
                'timestamp': self.timestamp,
                'run_directory': str(self.run_dir),
                'execution_time': datetime.now().isoformat()
            },
            'directory_structure': {k: str(v) for k, v in self.dirs.items()},
            'execution_log': self.pipeline_log,
            'results_summary': self.results
        }
        
        # Save to logs directory (create if not exists)
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        summary_file = logs_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save human-readable summary
        readable_file = logs_dir / "execution_summary.txt"
        with open(readable_file, 'w') as f:
            f.write("CARDIOAI PIPELINE EXECUTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Run ID: {self.timestamp}\n")
            f.write(f"Output Directory: {self.run_dir}\n")
            f.write(f"Execution Time: {datetime.now().isoformat()}\n\n")
            
            f.write("EXECUTION STEPS:\n")
            f.write("-" * 30 + "\n")
            for entry in self.pipeline_log:
                status_symbol = "OK" if entry['status'] == "SUCCESS" else "ERROR" if entry['status'] == "ERROR" else "RUNNING"
                duration = f" ({entry['duration_seconds']:.1f}s)" if entry.get('duration_seconds') else ""
                f.write(f"{status_symbol} {entry['step']}: {entry['status']}{duration}\n")
                if entry.get('details') and entry['status'] == "ERROR":
                    f.write(f"    Error: {entry['details']}\n")
            
            f.write(f"\nTotal Steps: {len(self.pipeline_log)}\n")
            successful_steps = sum(1 for entry in self.pipeline_log if entry['status'] == "SUCCESS")
            f.write(f"Successful Steps: {successful_steps}/{len(self.pipeline_log)}\n")
    
    def run_complete_pipeline(self):
        """Execute the complete enhanced CardioAI pipeline for all three aims"""
        print("ENHANCED CARDIOAI COMPLETE PIPELINE EXECUTION")
        print("=" * 60)
        print("=" * 60)
        print(f"Run ID: {self.timestamp}")
        print(f"Output Directory: {self.run_dir}")
        print(f"GPU Memory Configuration: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("  - All tasks: GPU 0")
        print("=" * 60)
        
        try:
            # GPU memory setup
            if torch.cuda.is_available():
                # Clear any existing GPU memory allocations
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU memory cleared at pipeline start")
                
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                print(f"GPU Available: {torch.cuda.get_device_name()}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Create directories
            self.create_directories()
            
            # Training or load weights
            if self.skip_train:
                print(f"\n{'='*50}")
                print("STEP 1: LOADING LATEST WEIGHTS (SKIPPING TRAINING)")
                print(f"{'='*50}")

                # Set training indices environment variable even when skipping training
                # This ensures ablation and other scripts can use the same configuration
                if self.train_indices:
                    os.environ['CARDIOAI_TRAIN_INDICES'] = self.train_indices
                    print(f"Using custom training indices: {self.train_indices}")
                else:
                    os.environ['CARDIOAI_TRAIN_INDICES'] = f"0-{self.train_size-1}"
                    print(f"Using default training indices: 0-{self.train_size-1} (patients 1-{self.train_size})")

                try:
                    latest_run = self.find_and_copy_latest_weights()
                    self.log_step("load_latest_weights", "SUCCESS", f"Loaded from {latest_run}")
                    print(f"OK load_latest_weights: SUCCESS")
                    print(f"Successfully loaded weights from run: {latest_run}")
                except Exception as e:
                    print(f"ERROR: Failed to load latest weights: {e}")
                    self.log_step("load_latest_weights", "ERROR", str(e))
                    return False
            else:
                print(f"\n{'='*50}")
                print("STEP 1: ENHANCED TRAINING")
                print(f"{'='*50}")
                print(f"Configuration: {self.epochs} epochs, {self.stage_epochs} stage epochs, batch size {self.batch_size}, {self.training_frames} frames")
                print(f"Training set: patients 1-{self.train_size} ({self.train_size} patients, {self.num_folds}-fold CV)")
                print(f"ViT unfreezing: {self.stages} layers (0=all frozen, 12=all unfrozen)")
                print("Features: Progressive unfreezing, discriminative learning rates, simplified attention modules, MSE loss")

                # Set environment variables for training configuration
                os.environ['CARDIOAI_EPOCHS'] = str(self.epochs)
                os.environ['CARDIOAI_STAGE_EPOCHS'] = str(self.stage_epochs)
                os.environ['CARDIOAI_BATCH_SIZE'] = str(self.batch_size)
                os.environ['CARDIOAI_TRAINING_FRAMES'] = str(self.training_frames)
                os.environ['CARDIOAI_TRAIN_SIZE'] = str(self.train_size)
                os.environ['CARDIOAI_STAGES'] = str(self.stages)
                os.environ['CARDIOAI_ABLATION_ATTENTIONS'] = str(self.ablation_attentions)
                os.environ['CARDIOAI_NUM_FOLDS'] = str(self.num_folds)
                os.environ['CARDIOAI_TIMESTAMP'] = self.timestamp

                # Pass custom training indices if specified
                if self.train_indices:
                    os.environ['CARDIOAI_TRAIN_INDICES'] = self.train_indices
                    print(f"Using custom training indices: {self.train_indices}")
                else:
                    # Default behavior: use train_size
                    os.environ['CARDIOAI_TRAIN_INDICES'] = f"0-{self.train_size-1}"
                
                success, stdout, stderr = self.run_script(
                    Path("train_cardioAI.py"), self.dirs['training']
                )
                if not success:
                    print(f"ERROR: Enhanced training failed. Cannot continue pipeline.")
                    print(f"Last error output: {stderr[-500:] if stderr else 'None'}")
                    return False
            
            # Check training results for summary
            training_correlations = self._check_training_correlations()
            
            # Check if training-only mode
            if self.training_only:
                print(f"\n{'='*50}")
                print("TRAINING-ONLY MODE: Skipping all subsequent steps")
                print(f"{'='*50}")
                self.save_pipeline_summary()
                return True
            
            # Attention ablation study
            if not self.skip_ablation:
                print(f"\n{'='*50}")
                print("STEP 2: ATTENTION ABLATION STUDY (AIM 1)")
                print(f"{'='*50}")
                print("Demonstrating enhancements of multistage attentions:")
                print("  - Spatial features from pre-trained ViT-Base (google/vit-base-patch16-224)")
                print("  - Temporal attention for aggregating dynamic cardiac motion across frames")
                print("  - Fusion attention for integrating multi-view information (FC/TC/SA/LA)")
                
                # Set reduced epochs for ablation study (configurable factor of main training epochs)
                ablation_epochs = max(1, int(self.epochs * self.ablation_factor))  # Minimum 1 epoch, configurable factor of main epochs
                os.environ['CARDIOAI_ABLATION_EPOCHS'] = str(ablation_epochs)
                os.environ['CARDIOAI_ABLATION_FACTOR'] = str(self.ablation_factor)
                os.environ['CARDIOAI_ABLATION_PATIENTS'] = str(self.ablation_patients)
                os.environ['CARDIOAI_ABLATION_BATCH_SIZE'] = str(self.ablation_batch_size)
                os.environ['CARDIOAI_TRAINING_FRAMES'] = str(self.training_frames)
                os.environ['CARDIOAI_NUM_FOLDS'] = str(self.num_folds)

                # Pass custom training indices if specified
                if self.train_indices:
                    os.environ['CARDIOAI_TRAIN_INDICES'] = self.train_indices
                else:
                    os.environ['CARDIOAI_TRAIN_INDICES'] = f"0-{self.train_size-1}"

                print(f"Ablation study will use {ablation_epochs} epochs ({self.ablation_factor}x of main {self.epochs} epochs) with {self.ablation_patients} patients and batch size {self.ablation_batch_size} for variants")
                
                success, stdout, stderr = self.run_script(
                    Path("attention_ablation_cardioAI.py"), self.dirs['ablation']
                )
                if not success:
                    print(f"WARNING: Attention ablation study failed, but continuing...")
                    print(f"Error: {stderr[-200:] if stderr else 'None'}")
            else:
                print(f"\n{'='*50}")
                print("STEP 2: ATTENTION ABLATION STUDY (AIM 1) - SKIPPED")
                print(f"{'='*50}")
                print("Skipping attention ablation study as requested")
                self.log_step("attention_ablation_cardioAI", "SKIPPED", "User requested skip")
            
            # Attention visualizations
            if not self.skip_visualizations:
                print(f"\n{'='*50}")
                print("STEP 3: ATTENTION VISUALIZATIONS GENERATION")
                print(f"{'='*50}")
                expected_viz = self.attention_frames * 4 * 9 * 3 * 2  # Dynamic calculation
                print("Generating comprehensive attention visualizations:")
                print("  - Temporal attention-rollout curves per view")
                print(f"  - {expected_viz} attention visualizations for patient E100017564")
                print(f"  - {self.attention_frames} frames x 4 views x 9 parameters x 3 formats x 2 assets")
                
                # Set environment variables for visualization step
                os.environ['CARDIOAI_ATTENTION_FRAMES'] = str(self.attention_frames)
                os.environ['CARDIOAI_TRAINING_FRAMES'] = str(self.training_frames)
                
                success, stdout, stderr = self.run_script(
                    Path("attention_visualizations_cardioAI.py"), self.dirs['attention_viz']
                )
                if not success:
                    print(f"ERROR: Attention visualizations generation failed.")
                    print(f"Error: {stderr[-300:] if stderr else 'None'}")
                
                # Verify visualizations
                print(f"\n{'='*50}")
                print("STEP 4: ATTENTION VISUALIZATION VERIFICATION")
                print(f"{'='*50}")
                verification_results = self.verify_attention_visualizations()
                self.results['attention_verification'] = verification_results
            else:
                print(f"\n{'='*50}")
                print("STEP 3-4: ATTENTION VISUALIZATIONS (AIM 2) - SKIPPED")
                print(f"{'='*50}")
                print("Skipping attention visualizations as requested")
                self.log_step("attention_visualizations_cardioAI", "SKIPPED", "User requested skip")
                # Set dummy verification results
                expected_total_dummy = self.attention_frames * 4 * 9 * 3 * 2 + 27
                verification_results = {'success': False, 'actual_total': 0, 'expected_total': expected_total_dummy}
                self.results['attention_verification'] = verification_results

            # Validation
            if self.run_validation:
                print(f"\n{'='*50}")
                print("STEP 5: INTERNAL VALIDATION")
                print(f"{'='*50}")
                print("Running comprehensive validation analysis:")
                print("  - Training curves (loss and correlation progress)")
                print("  - Correlation plots of true vs predicted values (patients 1-235 from All.xlsx)")
                print("  - Scatter and Bland-Altman plots")
                print("  - Confusion matrices")
                print("  - ROC curves with AUC values")
                print("  - UMAP/t-SNE embeddings with KMeans/DBSCAN clusters")
                print("  - Multi-sheet Excel comprehensive report")

                success, stdout, stderr = self.run_script(
                    Path("validation_cardioAI.py"), self.dirs['validation']
                )
                if not success:
                    print(f"WARNING: Validation failed, but continuing...")
                    print(f"Error: {stderr[-200:] if stderr else 'None'}")

            # Independent test
            if self.run_test:
                print(f"\n{'='*50}")
                print("STEP 6: TEST")
                print(f"{'='*50}")
                print("Running test on patients 236-308 from All.xlsx (73 patients)")
                print("  - Correlation plots of true vs predicted values")
                print("  - Scatter and Bland-Altman plots")
                print("  - Confusion matrices")
                print("  - ROC curves with AUC values")
                print("  - UMAP/t-SNE embeddings with KMeans/DBSCAN clusters")

                success, stdout, stderr = self.run_script(
                    Path("test_cardioAI.py"), self.dirs['test']
                )
                if not success:
                    print(f"WARNING: Test failed, but continuing...")
                    print(f"Error: {stderr[-200:] if stderr else 'None'}")

            # Final summary
            print(f"\n{'='*50}")
            print("STEP 7: FINAL SUMMARY GENERATION")
            print(f"{'='*50}")
            self.save_pipeline_summary()
            final_summary = self.generate_aims_summary()
            
            print(f"\n{'='*60}")
            print("ENHANCED PIPELINE EXECUTION COMPLETED")
            print(f"{'='*60}")
            print(f"Results saved to: {self.run_dir}")
            print(f"Total execution time: {sum(self.pipeline_log[i].get('duration_seconds', 0) or 0 for i in range(len(self.pipeline_log)))/3600:.2f} hours")
            
            # Final summary
            successful_steps = sum(1 for entry in self.pipeline_log if entry['status'] == "SUCCESS")
            total_steps = len(self.pipeline_log)
            print(f"Pipeline Success Rate: {successful_steps}/{total_steps} steps completed successfully")
            
            # Aims completion status
            print(f"\nAIMS COMPLETION STATUS:")
            print(f"  Aim 1 (Attention Ablation): {'OK Completed' if any('ablation' in entry['step'] and entry['status'] == 'SUCCESS' for entry in self.pipeline_log) else 'FAIL Failed'}")
            print(f"  Aim 2 (Visualizations): {'OK Completed' if verification_results.get('success', False) else 'FAIL Failed'}")
            
            expected_for_display = self.attention_frames * 4 * 9 * 3 * 2 + 27
            if verification_results.get('success', False):
                print(f"  Complete ({expected_for_display} Visualizations): OK Completed ({verification_results.get('actual_total', 0)}/{verification_results.get('expected_total', expected_for_display)})")
            else:
                print(f"  Complete ({expected_for_display} Visualizations): FAIL Incomplete ({verification_results.get('actual_total', 0)}/{verification_results.get('expected_total', expected_for_display)})")
            
            return True
            
        except Exception as e:
            print(f"FATAL ERROR in enhanced pipeline execution: {e}")
            traceback.print_exc()
            self.log_step("pipeline_execution", "FATAL_ERROR", str(e))
            return False


def main():
    """Main execution function"""
    import argparse
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='CardioAI Complete Pipeline')
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (increased for progressive ViT unfreezing)')
        parser.add_argument('--stage-epochs', type=int, default=50, help='Number of epochs per stage in progressive unfreezing')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
        parser.add_argument('--train-size', type=int, default=235, help='Number of patients for training (default: 235, patients 1-235)')
        parser.add_argument('--train-indices', type=str, default=None, help='Custom training indices as comma-separated ranges (e.g., "0-179,180-234"). Overrides --train-size if specified.')
        parser.add_argument('--num-folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
        parser.add_argument('--skip-train', action='store_true', help='Skip training and use latest weights for subsequent steps')
        parser.add_argument('--skip-ablation', action='store_true', help='Skip attention ablation study (Aim 1)')
        parser.add_argument('--skip-visualizations', action='store_true', help='Skip attention visualizations (Aim 2)')
        parser.add_argument('--training-only', action='store_true', help='Run only training step')
        parser.add_argument('--training-frames', type=int, default=32, help='Number of temporal frames for training')
        parser.add_argument('--frames', type=int, default=32, help='Number of frames with highest attention importance for visualizations')
        parser.add_argument('--ablation-factor', type=float, default=0.25, help='Factor to multiply training epochs for ablation studies')
        parser.add_argument('--ablation-patients', type=int, default=235, help='Number of patients for ablation studies (default: 235, patients 1-235)')
        parser.add_argument('--ablation-batch-size', type=int, default=16, help='Batch size for ablation study variants')
        parser.add_argument('--stages', type=int, default=1, help='Number of ViT layers to unfreeze progressively (0-12). 12 means all layers unfrozen for domain adaptation')
        parser.add_argument('--ablation_attentions', type=str, default='temporal,fusion', help='temporal,fusion. Use "none" for direct ViT->regression')

        # Validation flags - default to True (run by default)
        parser.add_argument('--run-validation', dest='run_validation', action='store_true', default=True, help='Run validation on patients 1-235 (235 patients, 5-fold CV cohort) [DEFAULT]')
        parser.add_argument('--no-validation', dest='run_validation', action='store_false', help='Skip validation')
        parser.add_argument('--run-test', dest='run_test', action='store_true', default=True, help='Run independent test on patients 236-308 (73 patients from All.xlsx) [DEFAULT]')
        parser.add_argument('--no-test', dest='run_test', action='store_false', help='Skip test')

        args = parser.parse_args()
        
        # Initialize pipeline
        pipeline = CardioAIPipeline()
        
        # Pass configuration to pipeline
        pipeline.epochs = args.epochs
        pipeline.stage_epochs = getattr(args, 'stage_epochs', 50)
        pipeline.batch_size = args.batch_size
        pipeline.training_frames = args.training_frames
        pipeline.train_size = getattr(args, 'train_size', 235)
        pipeline.train_indices = getattr(args, 'train_indices', None)
        pipeline.num_folds = getattr(args, 'num_folds', 5)
        pipeline.skip_train = getattr(args, 'skip_train', False)
        pipeline.skip_ablation = args.skip_ablation
        pipeline.skip_visualizations = args.skip_visualizations
        pipeline.training_only = args.training_only
        pipeline.attention_frames = args.frames
        pipeline.ablation_factor = getattr(args, 'ablation_factor', 0.25)
        pipeline.ablation_patients = getattr(args, 'ablation_patients', 235)
        pipeline.ablation_batch_size = getattr(args, 'ablation_batch_size', 16)
        pipeline.stages = args.stages
        pipeline.ablation_attentions = args.ablation_attentions
        pipeline.run_validation = getattr(args, 'run_validation', True)  # Default True
        pipeline.run_test = getattr(args, 'run_test', True)  # Default True
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\nSUCCESS: CardioAI pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nFAILURE: CardioAI pipeline encountered errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()