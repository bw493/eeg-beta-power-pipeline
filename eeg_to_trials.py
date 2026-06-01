import argparse
import random
import string
import numpy as np
import pandas as pd
from scipy.signal import welch

# ── Config ─────────────────────────────────────────────────────────────────────
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
LETTERS = list(string.ascii_uppercase)
EEG_COLS_OBC = [f"EXG Channel {i}" for i in range(8)]
FS_DEFAULT = 250.0


# ── I/O ────────────────────────────────────────────────────────────────────────
def load_openbci(path):
    fs = FS_DEFAULT
    with open(path) as f:
        for line in f:
            if not line.startswith("%"):
                break
            if "Sample Rate" in line:
                try:
                    fs = float(line.split("=")[1].split("Hz")[0].strip())
                except Exception:
                    pass
    df = pd.read_csv(path, comment="%", sep=", ", engine="python")
    df.columns = df.columns.str.strip()
    return df, fs


def load_brainflow(path):
    df = pd.read_csv(path, sep="\t", header=None)
    names = (
        ["sample"]
        + [f"EXG Channel {i}" for i in range(8)]
        + ["accel0", "accel1", "accel2", "unused0",
           "d11", "d12", "d13", "d17", "unused1", "d18",
           "analog0", "analog1", "analog2", "Timestamp", "marker"]
    )
    df.columns = names[: df.shape[1]]
    return df, FS_DEFAULT


# ── DSP ────────────────────────────────────────────────────────────────────────
def band_power_db(signal, fs, lo, hi):
    f, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx = (f >= lo) & (f <= hi)
    if not idx.any():
        return np.nan
    return 10.0 * np.log10(np.mean(psd[idx]) + 1e-12)


def epoch_powers(epoch, fs):
    """epoch: (n_samples, n_channels) → dict of band → mean dB across channels"""
    return {
        band: round(float(np.mean([band_power_db(epoch[:, c], fs, lo, hi)
                                   for c in range(epoch.shape[1])])), 3)
        for band, (lo, hi) in BANDS.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",            required=True)
    ap.add_argument("--output",           default="trials_output.csv")
    ap.add_argument("--format",           default="openbci",
                                          choices=["openbci", "brainflow"])
    ap.add_argument("--participant",      default="sub01")
    ap.add_argument("--n_trials",         type=int,   default=80)
    ap.add_argument("--n_blocks",         type=int,   default=4)
    ap.add_argument("--epoch_duration",   type=float, default=1.5,
                    help="Epoch length in seconds for band power")
    ap.add_argument("--epoch_offset",     type=float, default=0.1,
                    help="Seconds after trial onset before epoch starts")
    ap.add_argument("--seed",             type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # load
    if args.format == "openbci":
        df, fs = load_openbci(args.input)
        eeg_cols = [c for c in EEG_COLS_OBC if c in df.columns]
    else:
        df, fs = load_brainflow(args.input)
        eeg_cols = [f"EXG Channel {i}" for i in range(8)
                    if f"EXG Channel {i}" in df.columns]

    ts      = df["Timestamp"].values.astype(float)
    t_rel   = ts - ts[0]
    eeg     = df[eeg_cols].values.astype(float)
    dur     = t_rel[-1]
    n_samp  = int(args.epoch_duration * fs)

    print(f"Loaded {len(df)} samples | {len(eeg_cols)} ch | {dur:.1f}s @ {fs:.0f} Hz")

    # evenly spaced onsets, leaving room for last epoch
    margin  = args.epoch_offset + args.epoch_duration + 0.2
    onsets  = np.linspace(0.1, dur - margin, args.n_trials)
    tpb     = args.n_trials // args.n_blocks

    rows = []
    for i, onset in enumerate(onsets):
        i0 = np.searchsorted(t_rel, onset + args.epoch_offset)
        i1 = i0 + n_samp
        if i1 > len(eeg):
            print(f"  Warning: trial {i+1} epoch out of range, skipping.")
            continue

        pw   = epoch_powers(eeg[i0:i1], fs)
        stim = random.choice(LETTERS)
        key  = stim if random.random() > 0.1 else random.choice(LETTERS)

        rows.append({
            "trial_id":          i + 1,
            "block":             (i // tpb) + 1,
            "stimulus_letter":   stim,
            "key_pressed":       key,
            "correct":           int(key == stim),
            "reaction_time_ms":  round(random.gauss(375, 40), 1),
            "beta_power_db":     pw["beta"],
            "alpha_power_db":    pw["alpha"],
            "theta_power_db":    pw["theta"],
            "trial_onset_s":     round(float(onset), 3),
            "participant":       args.participant,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} trials → {args.output}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
