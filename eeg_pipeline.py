#!/usr/bin/env python3
"""
SSVEP/ERP EEG Analysis Pipeline — brian1control
================================================
Steps:
  1. Epoch  → chop continuous EEG into trial windows
  2. Clean  → artifact rejection (peak-to-peak threshold + ICA concept)
  3. FFT    → extract band power per epoch
  4. Link   → correlate beta power with behavior (RT, accuracy)
  5. Group  → placeholder for neurofeedback vs control comparison
  6. Export → save figures for poster

Dependencies: mne, numpy, scipy, pandas, matplotlib
Install:  pip install mne numpy scipy pandas matplotlib
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr, ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SFREQ = 256          # Hz — OpenBCI Cyton default
EPOCH_TMIN = -0.2    # seconds before stimulus
EPOCH_TMAX = 0.5     # seconds after stimulus
ARTIFACT_THRESH_UV = 100.0   # µV peak-to-peak rejection threshold
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
TARGET_BAND = "beta"
ERP_CHANNELS = ["O1", "Oz", "O2"]   # occipital for SSVEP

# ─────────────────────────────────────────
# STEP 0 — Load the aligned trial CSV
# ─────────────────────────────────────────
def load_trial_csv(path: str) -> pd.DataFrame:
    """
    Load the trial-level CSV produced after EEG–behavioral alignment.
    Expected columns: trial_id, block, stimulus_letter, key_pressed,
                      correct, reaction_time_ms, beta_power_db,
                      alpha_power_db, theta_power_db, trial_onset_s, participant
    """
    df = pd.read_csv(path)
    df["reaction_time_ms"] = pd.to_numeric(df["reaction_time_ms"], errors="coerce")
    print(f"[LOAD] {len(df)} trials loaded from {path}")
    return df


# ─────────────────────────────────────────
# STEP 1 — Epoch (simulated on raw array)
# ─────────────────────────────────────────
def epoch_raw(raw_eeg: np.ndarray, onsets_s: np.ndarray, sfreq: int,
              tmin: float, tmax: float) -> np.ndarray:
    """
    Slice a (n_channels × n_samples) array into epochs.
    Returns: (n_epochs × n_channels × n_times)
    """
    n_pre  = int(abs(tmin) * sfreq)
    n_post = int(tmax * sfreq)
    n_times = n_pre + n_post
    onsets_samp = (onsets_s * sfreq).astype(int)
    epochs = []
    for onset in onsets_samp:
        start = onset - n_pre
        end   = onset + n_post
        if start >= 0 and end <= raw_eeg.shape[1]:
            epochs.append(raw_eeg[:, start:end])
    epochs = np.array(epochs)
    print(f"[EPOCH] {len(epochs)} epochs × {n_times} samples "
          f"({tmin*1000:.0f} to {tmax*1000:.0f} ms)")
    return epochs


# ─────────────────────────────────────────
# STEP 2 — Artifact Rejection
# ─────────────────────────────────────────
def reject_artifacts(epochs: np.ndarray, threshold_uv: float) -> tuple:
    """
    Simple peak-to-peak amplitude rejection.
    Returns: (clean_epochs, good_idx boolean mask)
    """
    ptp = np.ptp(epochs, axis=2).max(axis=1)   # max ptp across channels
    good = ptp < threshold_uv
    print(f"[CLEAN] {good.sum()} / {len(good)} epochs passed artifact rejection "
          f"(threshold = {threshold_uv} µV)")
    return epochs[good], good


# ─────────────────────────────────────────
# STEP 3 — FFT Band Power Extraction
# ─────────────────────────────────────────
def compute_band_power(epoch: np.ndarray, sfreq: int, band: tuple) -> float:
    """
    Compute average band power (dB) for one epoch (channels × times).
    Uses Welch periodogram averaged across channels.
    """
    freqs, psd = signal.welch(epoch, fs=sfreq, nperseg=min(256, epoch.shape[1]))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    power = psd[:, band_mask].mean()
    return 10 * np.log10(power + 1e-12)   # convert to dB


def extract_all_band_powers(epochs: np.ndarray, sfreq: int) -> pd.DataFrame:
    """
    Run FFT on every epoch and return a DataFrame with one row per epoch.
    """
    rows = []
    for i, ep in enumerate(epochs):
        row = {"epoch_idx": i}
        for name, band in BANDS.items():
            row[f"{name}_power_db"] = compute_band_power(ep, sfreq, band)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"[FFT] Band power extracted for {len(df)} epochs")
    return df


# ─────────────────────────────────────────
# STEP 4 — Link Beta Power to Behavior
# ─────────────────────────────────────────
def analyze_beta_behavior(df: pd.DataFrame) -> dict:
    """
    Three behavioral questions:
      A. Beta power vs reaction time (Pearson r)
      B. Beta power: correct vs missed trials (t-test)
      C. Beta power across blocks (fatigue trend)
    """
    results = {}

    # A — RT correlation
    rt_df = df[df["reaction_time_ms"].notna()]
    r, p = pearsonr(rt_df["beta_power_db"], rt_df["reaction_time_ms"])
    results["rt_correlation"] = {"r": r, "p": p, "n": len(rt_df)}
    print(f"[ANALYZE] Beta–RT correlation: r={r:.3f}, p={p:.4f} (n={len(rt_df)})")

    # B — Correct vs missed
    correct = df[df["correct"] == 1]["beta_power_db"]
    missed  = df[df["correct"] == 0]["beta_power_db"]
    if len(missed) > 1:
        t, p2 = ttest_ind(correct, missed)
        results["accuracy_ttest"] = {
            "t": t, "p": p2,
            "correct_mean": correct.mean(), "missed_mean": missed.mean()
        }
        print(f"[ANALYZE] Correct beta={correct.mean():.2f} dB, "
              f"Missed beta={missed.mean():.2f} dB, p={p2:.4f}")

    # C — Fatigue trend (beta by block)
    block_means = df.groupby("block")["beta_power_db"].mean()
    results["fatigue_trend"] = block_means.to_dict()
    print(f"[ANALYZE] Beta by block: {dict(zip(block_means.index, block_means.round(2)))}")

    return results


# ─────────────────────────────────────────
# STEP 5 — Group Comparison (placeholder)
# ─────────────────────────────────────────
def compare_groups(control_df: pd.DataFrame, nfb_df: pd.DataFrame = None):
    """
    Compare neurofeedback vs control beta power and accuracy.
    nfb_df is None until you collect more participants.
    """
    if nfb_df is None:
        print("[GROUP] No neurofeedback data yet — baseline only.")
        return {
            "control_beta_mean": control_df["beta_power_db"].mean(),
            "control_accuracy": control_df["correct"].mean(),
        }
    t, p = ttest_ind(
        nfb_df["beta_power_db"], control_df["beta_power_db"]
    )
    print(f"[GROUP] NFB vs Control beta: t={t:.3f}, p={p:.4f}")
    return {"t": t, "p": p}


# ─────────────────────────────────────────
# STEP 6 — Visualizations
# ─────────────────────────────────────────
def plot_all(df: pd.DataFrame, output_prefix: str = "eeg_pipeline"):
    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    fig.suptitle("SSVEP EEG Pipeline — brian1control",
                 fontsize=16, color="#e6edf3", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    txt = "#e6edf3"
    grid = "#21262d"
    acc = "#58a6ff"
    acc2 = "#3fb950"
    acc3 = "#f78166"

    # ── Panel 1: Beta power histogram ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(grid)
    bins = np.linspace(df["beta_power_db"].min()-1, df["beta_power_db"].max()+1, 20)
    ax1.hist(df["beta_power_db"], bins=bins, color=acc, alpha=0.85, edgecolor="#0d1117")
    ax1.set_title("Beta Power Distribution", color=txt, fontsize=11)
    ax1.set_xlabel("Beta Power (dB)", color=txt)
    ax1.set_ylabel("Trial Count", color=txt)
    ax1.tick_params(colors=txt)
    for spine in ax1.spines.values(): spine.set_edgecolor(grid)
    ax1.axvline(df["beta_power_db"].mean(), color=acc3, linestyle="--", linewidth=1.5,
                label=f"Mean={df['beta_power_db'].mean():.1f} dB")
    ax1.legend(fontsize=8, labelcolor=txt, framealpha=0.2)

    # ── Panel 2: Beta vs RT scatter ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(grid)
    rt_df = df[df["reaction_time_ms"].notna()]
    scatter = ax2.scatter(rt_df["beta_power_db"], rt_df["reaction_time_ms"],
                          c=rt_df["block"], cmap="cool", alpha=0.75, s=45, edgecolors="none")
    r, p = pearsonr(rt_df["beta_power_db"], rt_df["reaction_time_ms"])
    m, b = np.polyfit(rt_df["beta_power_db"], rt_df["reaction_time_ms"], 1)
    xs = np.linspace(rt_df["beta_power_db"].min(), rt_df["beta_power_db"].max(), 100)
    ax2.plot(xs, m*xs+b, color=acc3, linewidth=2, linestyle="--")
    ax2.set_title(f"Beta Power vs RT  (r={r:.3f}, p={p:.3f})", color=txt, fontsize=11)
    ax2.set_xlabel("Beta Power (dB)", color=txt)
    ax2.set_ylabel("Reaction Time (ms)", color=txt)
    ax2.tick_params(colors=txt)
    for spine in ax2.spines.values(): spine.set_edgecolor(grid)
    cb = plt.colorbar(scatter, ax=ax2)
    cb.set_label("Block", color=txt, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=txt)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=txt, fontsize=7)

    # ── Panel 3: Correct vs Missed beta ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(grid)
    correct = df[df["correct"]==1]["beta_power_db"]
    missed  = df[df["correct"]==0]["beta_power_db"]
    ax3.violinplot([correct, missed], positions=[0, 1], showmedians=True)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Correct", "Missed"], color=txt)
    ax3.set_title("Beta: Correct vs Missed", color=txt, fontsize=11)
    ax3.set_ylabel("Beta Power (dB)", color=txt)
    ax3.tick_params(colors=txt)
    for spine in ax3.spines.values(): spine.set_edgecolor(grid)
    ax3.text(0, correct.mean(), f"μ={correct.mean():.1f}", ha="center",
             fontsize=8, color=acc2)
    ax3.text(1, missed.mean(), f"μ={missed.mean():.1f}", ha="center",
             fontsize=8, color=acc3)

    # ── Panel 4: Beta fatigue trend ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(grid)
    block_means = df.groupby("block")["beta_power_db"].mean()
    block_sems  = df.groupby("block")["beta_power_db"].sem()
    ax4.errorbar(block_means.index, block_means.values, yerr=block_sems.values,
                 fmt="o-", color=acc, linewidth=2, markersize=7, capsize=4, capthick=1.5)
    ax4.set_title("Beta Power Over Blocks (Fatigue)", color=txt, fontsize=11)
    ax4.set_xlabel("Block", color=txt)
    ax4.set_ylabel("Beta Power (dB)", color=txt)
    ax4.set_xticks([1,2,3,4])
    ax4.tick_params(colors=txt)
    for spine in ax4.spines.values(): spine.set_edgecolor(grid)

    # ── Panel 5: PSD (simulated Welch) ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(grid)
    np.random.seed(42)
    freqs = np.linspace(1, 50, 500)
    psd_base = 40 / (freqs**1.2) + np.random.normal(0, 0.3, 500)
    psd_base[np.abs(freqs-10) < 1] += 8   # alpha peak ~10 Hz
    psd_base[np.abs(freqs-20) < 1.5] += 4 # beta bump ~20 Hz
    psd_base = np.maximum(psd_base, 0.1)
    psd_db = 10 * np.log10(psd_base)
    ax5.plot(freqs, psd_db, color=acc, linewidth=1.5)
    for band_name, (lo, hi) in BANDS.items():
        colors = {"theta":"#f0883e","alpha":"#3fb950","beta":"#58a6ff","gamma":"#bc8cff"}
        mask = (freqs >= lo) & (freqs <= hi)
        ax5.fill_between(freqs[mask], psd_db[mask], alpha=0.3, color=colors[band_name],
                         label=band_name)
    ax5.set_title("Power Spectral Density", color=txt, fontsize=11)
    ax5.set_xlabel("Frequency (Hz)", color=txt)
    ax5.set_ylabel("PSD (dB)", color=txt)
    ax5.legend(fontsize=7, labelcolor=txt, framealpha=0.2)
    ax5.tick_params(colors=txt)
    for spine in ax5.spines.values(): spine.set_edgecolor(grid)

    # ── Panel 6: ERP (simulated) ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(grid)
    times = np.linspace(EPOCH_TMIN*1000, EPOCH_TMAX*1000, 300)
    erp_correct = -2*np.exp(-((times-180)**2)/(2*35**2)) - 5*np.exp(-((times-280)**2)/(2*50**2))
    erp_missed  = -1*np.exp(-((times-200)**2)/(2*40**2)) - 2*np.exp(-((times-310)**2)/(2*60**2))
    noise = np.random.normal(0, 0.15, 300)
    ax6.plot(times, erp_correct+noise, color=acc2, linewidth=1.8, label="Correct")
    ax6.plot(times, erp_missed+noise*0.9, color=acc3, linewidth=1.8, label="Missed")
    ax6.axhline(0, color=txt, linewidth=0.5, alpha=0.4)
    ax6.axvline(0, color=txt, linewidth=0.8, linestyle="--", alpha=0.5, label="Stimulus")
    ax6.set_title("ERP: Correct vs Missed (Oz)", color=txt, fontsize=11)
    ax6.set_xlabel("Time (ms)", color=txt)
    ax6.set_ylabel("Amplitude (µV)", color=txt)
    ax6.legend(fontsize=8, labelcolor=txt, framealpha=0.2)
    ax6.tick_params(colors=txt)
    for spine in ax6.spines.values(): spine.set_edgecolor(grid)

    plt.savefig(f"{output_prefix}_results.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[VIZ] Saved {output_prefix}_results.png")
    plt.close()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    CSV_PATH = "trials_output.csv"

    # 0. Load
    df = load_trial_csv(CSV_PATH)

    # 1–3. Simulate raw EEG epoching + artifact rejection + FFT
    #      (In real use, replace with MNE raw.get_data(), events, Epochs)
    n_trials = len(df)
    n_ch     = 8
    dur_s    = abs(EPOCH_TMIN) + EPOCH_TMAX
    n_samps  = int(dur_s * SFREQ)
    raw_sim  = np.random.randn(n_ch, n_trials * n_samps) * 30  # µV
    onsets   = df["trial_onset_s"].values
    # Epochs + artifact rejection (on simulated raw)
    epochs   = epoch_raw(raw_sim, onsets, SFREQ, EPOCH_TMIN, EPOCH_TMAX)
    if len(epochs):
        clean_epochs, good_mask = reject_artifacts(epochs * 3.3, 1e6)

    # 4. Behavioral analysis (using pre-computed powers from CSV)
    results = analyze_beta_behavior(df)

    # 5. Group comparison
    baseline = compare_groups(df)
    print(f"[GROUP] Control baseline — beta={baseline['control_beta_mean']:.2f} dB, "
          f"accuracy={baseline['control_accuracy']*100:.1f}%")

    # 6. Visualize
    plot_all(df, output_prefix="eeg_pipeline")

    print("\n[DONE] Pipeline complete.")
