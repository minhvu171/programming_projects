"""
This script loads, cleans, and analyzes ECG time-series data from a CSV file.

It preprocesses raw and pre-filtered ECG signals and visualizes signals in both 
the time and frequency domains. The script applies bandpass and notch filtering, 
compares filtering effects using power spectral density plots, and performs ECG 
waveform delineation (P, Q, R, S, T peaks) using NeuroKit2. It is intended for 
exploratory ECG signal quality assessment and feature-level inspection.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk


# Configuration
FS = 500  # Sampling rate (Hz)
DATA_PATH = Path("samples.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# Data loading and preprocessing
def load_and_clean_ecg_data(file_path: Path) -> pd.DataFrame:
    try:
        logging.info("Loading ECG data from %s", file_path)

        df = pd.read_csv(file_path, header=None, skiprows=1)
        df.columns = ["Time", "ECG_I_Unfiltered", "ECG_I_Filtered"]

        # Drop unit row
        df = df.iloc[1:].copy()

        # Clean time column
        df["Time"] = df["Time"].astype(str).str.strip("'")

        # Convert ECG columns to numeric
        df["ECG_I_Unfiltered"] = pd.to_numeric(df["ECG_I_Unfiltered"], errors="coerce")
        df["ECG_I_Filtered"] = pd.to_numeric(df["ECG_I_Filtered"], errors="coerce")

        df = df.dropna(subset=["ECG_I_Unfiltered", "ECG_I_Filtered"])

        # Convert time to seconds
        df["Time_s"] = df["Time"].apply(time_to_seconds)
        df = df.dropna(subset=["Time_s"])

        logging.info("Final dataset shape: %s", df.shape)
        return df
    except FileNotFoundError:
        logging.error("ECG data file not found: %s", file_path)
        raise


def time_to_seconds(time_str: str):
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            minutes, s_ms = parts
            hours = 0
        elif len(parts) == 3:
            hours, minutes, s_ms = parts
        else:
            return None

        seconds, milliseconds = s_ms.split(".")
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(milliseconds) / 1000
        )
    except Exception:
        return None


# Plotting utilities
def plot_raw_vs_filtered(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Time_s"], df["ECG_I_Unfiltered"], label="Unfiltered", alpha=0.6)
    plt.plot(df["Time_s"], df["ECG_I_Filtered"], label="Filtered")
    plt.title("ECG Signal: Unfiltered vs Filtered")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_psd(signal_raw, signal_filt):
    f_raw, psd_raw = signal.welch(signal_raw, fs=FS, nperseg=1024)
    f_filt, psd_filt = signal.welch(signal_filt, fs=FS, nperseg=1024)

    plt.figure(figsize=(10, 6))
    plt.semilogy(f_raw, psd_raw, label="Unfiltered")
    plt.semilogy(f_filt, psd_filt, label="Filtered")
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (V²/Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (f_raw, psd_raw), (f_filt, psd_filt)


# ECG analysis
def plot_ecg_with_detected_peaks(ecg_signal, time_s, label: str):
    _, ecg_info = nk.ecg_process(ecg_signal, sampling_rate=FS)

    _, delineate_info = nk.ecg_delineate(
        ecg_signal,
        rpeaks=ecg_info["ECG_R_Peaks"],
        sampling_rate=FS,
        method="dwt",
        show=True,
        show_type="peaks"
    )

    plt.title(f"{label} delineated ECG complexes")
    plt.show()

    plt.figure(figsize=(20, 6))
    plt.plot(time_s, ecg_signal, color="gray", label="ECG")

    plt.scatter(time_s.iloc[ecg_info["ECG_R_Peaks"]],
                ecg_signal[ecg_info["ECG_R_Peaks"]],
                label="R", s=40,color='cyan')

    for wave, color in [
        ("ECG_P_Peaks", "blue"),
        ("ECG_Q_Peaks", "orange"),
        ("ECG_S_Peaks", "green"),
        ("ECG_T_Peaks", "red"),
    ]:
        plt.scatter(
            time_s.iloc[delineate_info[wave]],
            ecg_signal[delineate_info[wave]],
            label=wave.split("_")[1],
            s=30,
            color=color
        )

    plt.title(f"{label} ECG with detected waveforms")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Filtering pipeline
def apply_bandpass_and_notch(ecg_signal):
    # Bandpass
    b_bp, a_bp = signal.butter(
        4,
        [0.01 / (FS / 2), 150 / (FS / 2)],
        btype="band"
    )
    ecg_bp = signal.filtfilt(b_bp, a_bp, ecg_signal)

    # Notch
    b_notch, a_notch = signal.iirnotch(50 / (FS / 2), 30)
    ecg_notch = signal.filtfilt(b_notch, a_notch, ecg_bp)

    return ecg_bp, ecg_notch


def plot_filtering_summary(time_s, raw, bp, notch):
    signals = [
        ("Raw ECG", raw),
        ("Bandpass ECG", bp),
        ("Bandpass + Notch ECG", notch)
    ]

    plt.figure(figsize=(15, 26))
    for i, (title, sig) in enumerate(signals, start=1):
        plt.subplot(3, 2, 2*i - 1)
        plt.plot(time_s, sig)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.grid(True)

        f, psd = signal.welch(sig, fs=FS, nperseg=1024)
        plt.subplot(3, 2, 2*i)
        plt.semilogy(f, psd)
        plt.title(f"PSD: {title}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (V²/Hz)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main execution
def main():
    df = load_and_clean_ecg_data(DATA_PATH)

    ecg_raw = df["ECG_I_Unfiltered"].values
    ecg_filt = df["ECG_I_Filtered"].values

    plot_raw_vs_filtered(df)
    plot_psd(ecg_raw, ecg_filt)

    plot_ecg_with_detected_peaks(ecg_raw, df["Time_s"], "Raw")
    plot_ecg_with_detected_peaks(ecg_filt, df["Time_s"], "Filtered")

    ecg_bp, ecg_notch = apply_bandpass_and_notch(ecg_raw)
    plot_filtering_summary(df["Time_s"], ecg_raw, ecg_bp, ecg_notch)


if __name__ == "__main__":
    main()
