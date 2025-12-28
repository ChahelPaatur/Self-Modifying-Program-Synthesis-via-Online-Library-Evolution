## Figures

### Figure 1: Train-score distribution
File: `paper_assets/figures/fig_train_score_hist.pdf` (and `.png`)

**Caption.** Distribution of average train-set similarity scores obtained by SMPMA on the ARC-AGI evaluation challenges. Higher scores indicate better fit on training pairs but do not necessarily imply strict generalization to test cases.

### Figure 2: Strict accuracy
File: `paper_assets/figures/fig_strict_accuracy.pdf` (and `.png`)

**Caption.** Strict task accuracy on the ARC-AGI evaluation challenges, where a task is counted as solved only if the synthesized program produces correct outputs for all test cases in that task.

### Figure 3: Runtime distribution
File: `paper_assets/figures/fig_runtime_hist.pdf` (and `.png`)

**Caption.** Distribution of per-task runtimes for SMPMA (seconds per task) on the ARC-AGI evaluation challenges under the current hyperparameters.

### Figure 4: Program length distribution
File: `paper_assets/figures/fig_program_len_hist.pdf` (and `.png`)

**Caption.** Distribution of synthesized program lengths (number of operations) across evaluation tasks.

### Figure 5: Ablation accuracy (ARC-AGI subset)
File: `paper_assets/figures/fig_ablation_accuracy.pdf` (and `.png`)

**Caption.** Ablation study on an ARC-AGI evaluation subset comparing strict accuracy under toggles such as macro-templates, operation prioritization, and scoring mode. This plot is intended to isolate the contribution of individual mechanisms.

### Figure 6: AeroSynth accuracy (aerospace-motivated synthetic benchmark)
File: `paper_assets/figures/fig_aero_accuracy.pdf` (and `.png`)

**Caption.** Strict accuracy on AeroSynth, a controlled synthetic benchmark capturing aerospace-relevant perception and autonomy motifs (target extraction, translation/tracking, and compositional overlay).

### Figure 7: Spacecraft navigation ablation success rate
File: `paper_assets/figures/fig_spacecraft_ablation_success.pdf` (and `.png`)

**Caption.** Success rate (20 random seeds) for autonomous spacecraft rendezvous/docking under controller and safety-stack ablations. Variants compare LQR vs sampling-MPC, docking-mode switching, and a safety shield.

### Figure 8: Spacecraft navigation ablation fuel usage
File: `paper_assets/figures/fig_spacecraft_ablation_fuel.pdf` (and `.png`)

**Caption.** Fuel usage distribution (delta-v proxy) for spacecraft rendezvous/docking across ablation variants (20 random seeds). Lower is better for comparable success.
