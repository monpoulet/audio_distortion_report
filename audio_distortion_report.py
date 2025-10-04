#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_distortion_report_mc.py

Multi-channel, multithreaded distortion report between a reference ("before")
and a processed ("after") audio file.

Per-channel it computes:
  - Alignment (shared shift determined on mono mix for stability)
  - Gain matching
  - Residual & SDR
  - Spectral FFT
  - THD / SINAD / ENOB (if single-tone; or with --f0 hint)

Also computes an overall mono summary.

Author: Marc André a.k.a. Agent X27, 2025.
"""
import argparse
import os
import math
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional high-precision resampler
try:
    import soxr  # python-soxr
    HAVE_SOXR = True
except Exception:
    HAVE_SOXR = False


def read_audio(path: str):
    """Read audio as float64, shape (N, C), return (data, sr)."""
    data, sr = sf.read(path, always_2d=True)
    data = data.astype(np.float64, copy=False)
    return data, sr


def precise_resample(x, sr_in, sr_out):
    """High-quality resample x (shape (N, C)) from sr_in to sr_out (float64)."""
    if sr_in == sr_out:
        return x
    if HAVE_SOXR:
        # soxr can resample multi-channel arrays directly
        y = soxr.resample(x, sr_in, sr_out, quality='HQ', channels=x.shape[1])
        return y.astype(np.float64)
    # Fallback: apply resample_poly per channel
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    chans = []
    for c in range(x.shape[1]):
        chans.append(signal.resample_poly(x[:, c], up, down).astype(np.float64))
    # Re-pack
    m = min(len(ch) for ch in chans)
    y = np.stack([ch[:m] for ch in chans], axis=1)
    return y


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(x**2)) if x.size else np.nan


def align_signals_mono(x_mono, y_mono, sr, max_shift_s=0.1):
    """
    Determine a single global shift between reference mono x_mono and processed mono y_mono,
    using cross-correlation (on a downsampled proxy). Returns shift_samples (y shifted forward if positive).
    """
    max_shift = int(max_shift_s * sr)
    if max_shift < 1:
        return 0

    ds = max(1, sr // 4000)
    x_ds = x_mono[::ds]
    y_ds = y_mono[::ds]
    n = min(len(x_ds), len(y_ds))
    x_ds = x_ds[:n]
    y_ds = y_ds[:n]

    corr = signal.correlate(y_ds, x_ds, mode='full', method='auto')
    lags = np.arange(-len(x_ds) + 1, len(y_ds))
    lag_ds = lags[np.argmax(corr)]
    lag_full = int(lag_ds * ds)
    lag_full = max(-max_shift, min(max_shift, lag_full))
    return lag_full


def apply_shift(y, shift):
    """Shift a multi-channel signal by 'shift' samples (y delayed vs x if shift>0). Trim/pad to preserve alignment window."""
    if shift > 0:
        y_shifted = y[shift:, :]
    elif shift < 0:
        y_shifted = np.pad(y, ((abs(shift), 0), (0, 0)), mode='constant')
    else:
        y_shifted = y
    return y_shifted


def optimal_gain_match(x, y):
    """
    Least-squares gain to scale y ~ g*x. Returns g.
    If denominator ~0, fall back to 1.0
    """
    denom = np.dot(x, x)
    if denom <= 1e-20:
        return 1.0
    g = np.dot(x, y) / denom
    return g


def compute_fft_metrics(x, sr, f0_hint=None, search_bw_hz=5.0, thd_harmonics=10):
    """
    Compute single-tone metrics from x assuming it *contains* a sine.
    Returns dict with f0, Pfund, sumPharm, THD, SINAD, ENOB and a spectrum for plotting.
    If no clear tone is found and f0_hint is None, returns minimal info.
    """
    N = 2**int(np.ceil(np.log2(len(x))))
    w = get_window('hann', len(x), fftbins=True)
    xw = np.zeros(N, dtype=np.float64)
    xw[:len(x)] = x * w
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    P = (np.abs(X)**2) / (np.sum(w**2))

    def nearest_bin(freq):
        return int(np.round(freq / (sr / N)))

    if f0_hint is None:
        min_bin = max(1, int(20.0 / (sr / N)))
        k0 = np.argmax(P[min_bin:]) + min_bin
    else:
        k0 = nearest_bin(f0_hint)
        k0 = max(1, min(k0, len(P)-1))

    f0 = freqs[k0]
    Pfund = P[k0]

    dominant_ok = Pfund > 10.0 * np.median(P[max(1, k0-50):k0+51])

    def band_power(center_freq):
        half = int(np.ceil(search_bw_hz / (sr / N)))
        kc = int(np.round(center_freq / (sr / N)))
        k1 = max(0, kc - half)
        k2 = min(len(P)-1, kc + half)
        return np.sum(P[k1:k2+1])

    Pfund_band = band_power(f0)

    Pharm_sum = 0.0
    for h in range(2, thd_harmonics+1):
        fh = h * f0
        if fh >= sr/2:
            break
        Pharm_sum += band_power(fh)

    THD = np.sqrt(Pharm_sum) / np.sqrt(max(Pfund_band, 1e-300))

    Ptotal = np.sum(P)
    Pnoise_dist = max(Ptotal - Pfund_band, 1e-300)
    SINAD = 10.0 * np.log10(max(Pfund_band, 1e-300) / Pnoise_dist)
    ENOB = (SINAD - 1.76) / 6.02

    return {
        "f0_hz": float(f0),
        "fund_power": float(Pfund_band),
        "harm_power_sum": float(Pharm_sum),
        "THD": float(THD),
        "SINAD_dB": float(SINAD),
        "ENOB_bits": float(ENOB),
        "dominant_tone_detected": bool(dominant_ok),
        "freqs": freqs,   # numpy arrays (we won't store in JSON per channel to keep size down)
        "power": P,
    }


def save_spectrum_plot(freqs, power, out_png, title, xlim_hz=None):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(freqs, 10*np.log10(np.maximum(power, 1e-300)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB, arbitrary)")
    plt.title(title)
    if xlim_hz:
        plt.xlim(0, xlim_hz)
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def analyze_channel(idx, x_ch, y_ch, sr, outdir, f0_hint=None, fft_xlim=None):
    """
    Per-channel analysis. Returns dict of metrics and plot paths.
    """
    # Gain match per channel
    g = optimal_gain_match(x_ch, y_ch)
    y_matched = y_ch * g
    residual = y_matched - x_ch

    s_power = np.sum(x_ch**2) + 1e-300
    d_power = np.sum(residual**2) + 1e-300
    SDR = 10.0 * np.log10(s_power / d_power)

    # FFT metrics
    fft_after = compute_fft_metrics(y_matched, sr, f0_hint=f0_hint)
    fft_before = compute_fft_metrics(x_ch, sr, f0_hint=f0_hint)
    fft_resid = compute_fft_metrics(residual, sr, f0_hint=fft_after.get("f0_hz", None))

    # Plots
    base = os.path.join(outdir, f"ch{idx:02d}")
    os.makedirs(base, exist_ok=True)

    spec_before_png = os.path.join(base, "spectrum_before.png")
    spec_after_png = os.path.join(base, "spectrum_after.png")
    spec_resid_png = os.path.join(base, "spectrum_residual.png")

    save_spectrum_plot(fft_before["freqs"], fft_before["power"], spec_before_png,
                       f"Channel {idx} - BEFORE")
    save_spectrum_plot(fft_after["freqs"], fft_after["power"], spec_after_png,
                       f"Channel {idx} - AFTER")
    save_spectrum_plot(fft_resid["freqs"], fft_resid["power"], spec_resid_png,
                       f"Channel {idx} - RESIDUAL")

    if fft_xlim:
        save_spectrum_plot(fft_before["freqs"], fft_before["power"],
                           os.path.join(base, "spectrum_before_zoom.png"),
                           f"Channel {idx} - BEFORE 0..{fft_xlim} Hz", xlim_hz=fft_xlim)
        save_spectrum_plot(fft_after["freqs"], fft_after["power"],
                           os.path.join(base, "spectrum_after_zoom.png"),
                           f"Channel {idx} - AFTER 0..{fft_xlim} Hz", xlim_hz=fft_xlim)
        save_spectrum_plot(fft_resid["freqs"], fft_resid["power"],
                           os.path.join(base, "spectrum_residual_zoom.png"),
                           f"Channel {idx} - RESIDUAL 0..{fft_xlim} Hz", xlim_hz=fft_xlim)

    thd = fft_after["THD"]
    sinad = fft_after["SINAD_dB"]
    enob = fft_after["ENOB_bits"]
    f0 = fft_after["f0_hz"]
    dominant = fft_after["dominant_tone_detected"]

    ch_report = {
        "channel_index": idx,
        "gain_match": float(g),
        "SDR_dB": float(SDR),
        "fundamental_hz": float(f0),
        "THD_ratio": float(thd),
        "THD_percent": float(100.0*thd),
        "SINAD_dB": float(sinad),
        "ENOB_bits": float(enob),
        "dominant_tone_detected": bool(dominant),
        "plots": {
            "before": os.path.abspath(spec_before_png),
            "after": os.path.abspath(spec_after_png),
            "residual": os.path.abspath(spec_resid_png),
        }
    }
    return ch_report


def main():
    ap = argparse.ArgumentParser(description="Multi-channel, multithreaded audio distortion report.")
    ap.add_argument("before", type=str, help="Reference audio file (before)")
    ap.add_argument("after", type=str, help="Processed audio file (after)")
    ap.add_argument("--out", type=str, default="distortion_report_mc", help="Output directory")
    ap.add_argument("--f0", type=float, default=None, help="Fundamental frequency hint in Hz (for THD/SINAD)")
    ap.add_argument("--max_shift_ms", type=float, default=100.0, help="Max alignment shift (ms) to search (global, mono-based)")
    ap.add_argument("--fft_xlim", type=float, default=None, help="Optional x-axis max for spectrum plots (Hz)")
    ap.add_argument("--workers", type=int, default=None, help="Max worker threads (default: min(8, 3*C))")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X, sr_x = read_audio(args.before)   # (N, Cx)
    Y, sr_y = read_audio(args.after)    # (M, Cy)

    # Resample AFTER to match BEFORE
    Y_rs = precise_resample(Y, sr_y, sr_x)

    # Compute global shift on mono downmix (equal-power)
    x_mono = np.mean(X, axis=1)
    y_mono = np.mean(Y_rs, axis=1)
    shift = align_signals_mono(x_mono, y_mono, sr_x, max_shift_s=args.max_shift_ms/1000.0)

    # Apply single global shift to all channels
    Y_shifted = apply_shift(Y_rs, shift)

    # Time trim to overlap with BEFORE
    n = min(len(X), len(Y_shifted))
    X = X[:n, :]
    Y_shifted = Y_shifted[:n, :]

    Cx = X.shape[1]
    Cy = Y_shifted.shape[1]
    C = min(Cx, Cy)  # analyze up to min channel count
    if C == 0:
        raise RuntimeError("No overlapping channels to analyze.")

    # Overall mono summary (after channel trim)
    x_mono = np.mean(X[:, :C], axis=1)
    y_mono = np.mean(Y_shifted[:, :C], axis=1)

    # Per-channel analysis in parallel
    max_workers = args.workers if args.workers is not None else min(8, 3*C)
    results = [None]*C
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for ch in range(C):
            futures[ex.submit(analyze_channel, ch, X[:, ch], Y_shifted[:, ch], sr_x,
                              args.out, args.f0, args.fft_xlim)] = ch
        for fut in as_completed(futures):
            ch = futures[fut]
            results[ch] = fut.result()

    # Mono summary metrics (same pipeline as a channel, but single pass, not parallel)
    g_mono = optimal_gain_match(x_mono, y_mono)
    y_mono_matched = y_mono * g_mono
    resid_mono = y_mono_matched - x_mono
    s_power = np.sum(x_mono**2) + 1e-300
    d_power = np.sum(resid_mono**2) + 1e-300
    SDR_mono = 10.0 * np.log10(s_power / d_power)

    fft_after_m = compute_fft_metrics(y_mono_matched, sr_x, f0_hint=args.f0)
    fft_before_m = compute_fft_metrics(x_mono, sr_x, f0_hint=args.f0)
    fft_resid_m = compute_fft_metrics(resid_mono, sr_x, f0_hint=fft_after_m.get("f0_hz", None))

    # Mono plots
    spec_before_png = os.path.join(args.out, "mono_spectrum_before.png")
    spec_after_png = os.path.join(args.out, "mono_spectrum_after.png")
    spec_resid_png = os.path.join(args.out, "mono_spectrum_residual.png")
    save_spectrum_plot(fft_before_m["freqs"], fft_before_m["power"], spec_before_png, "MONO - BEFORE")
    save_spectrum_plot(fft_after_m["freqs"], fft_after_m["power"], spec_after_png, "MONO - AFTER")
    save_spectrum_plot(fft_resid_m["freqs"], fft_resid_m["power"], spec_resid_png, "MONO - RESIDUAL")

    if args.fft_xlim:
        save_spectrum_plot(fft_before_m["freqs"], fft_before_m["power"],
                           os.path.join(args.out, "mono_spectrum_before_zoom.png"),
                           f"MONO - BEFORE 0..{args.fft_xlim} Hz", xlim_hz=args.fft_xlim)
        save_spectrum_plot(fft_after_m["freqs"], fft_after_m["power"],
                           os.path.join(args.out, "mono_spectrum_after_zoom.png"),
                           f"MONO - AFTER 0..{args.fft_xlim} Hz", xlim_hz=args.fft_xlim)
        save_spectrum_plot(fft_resid_m["freqs"], fft_resid_m["power"],
                           os.path.join(args.out, "mono_spectrum_residual_zoom.png"),
                           f"MONO - RESIDUAL 0..{args.fft_xlim} Hz", xlim_hz=args.fft_xlim)

    thd_m = fft_after_m["THD"]
    sinad_m = fft_after_m["SINAD_dB"]
    enob_m = fft_after_m["ENOB_bits"]
    f0_m = fft_after_m["f0_hz"]
    dominant_m = fft_after_m["dominant_tone_detected"]

    # Build report JSON
    report = {
        "files": {"before": args.before, "after": args.after},
        "sample_rate_hz": sr_x,
        "channels": {"before": X.shape[1], "after": Y_shifted.shape[1], "analyzed": C},
        "processing": {
            "resampler": "soxr(HQ)" if HAVE_SOXR else "scipy.signal.resample_poly",
            "alignment_shift_samples": int(shift),
            "alignment_shift_ms": float(1000.0*shift/sr_x),
            "gain_match_mono": float(g_mono),
            "workers": max_workers,
        },
        "mono_metrics": {
            "SDR_dB": float(SDR_mono),
            "fundamental_hz": float(f0_m),
            "THD_ratio": float(thd_m),
            "THD_percent": float(100.0*thd_m),
            "SINAD_dB": float(sinad_m),
            "ENOB_bits": float(enob_m),
            "dominant_tone_detected": bool(dominant_m),
            "plots": {
                "before": os.path.abspath(spec_before_png),
                "after": os.path.abspath(spec_after_png),
                "residual": os.path.abspath(spec_resid_png),
            }
        },
        "per_channel": results,
        "notes": [
            "A single global time shift is estimated on mono for robustness, then applied to all channels.",
            "Per-channel gain is matched independently (least squares).",
            "THD/SINAD/ENOB are most meaningful for single-tone content; use --f0 to force analysis if needed."
        ]
    }

    # Save JSON + Markdown
    json_path = os.path.join(args.out, "distortion_report_mc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(args.out, "distortion_report_mc.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Channel Audio Distortion Report\n\n")
        f.write(f"**Before:** {args.before}\n\n")
        f.write(f"**After:** {args.after}\n\n")
        f.write(f"- Sample rate (analysis): **{sr_x} Hz**\n")
        f.write(f"- Channels (before/after/analyzed): **{X.shape[1]}/{Y_shifted.shape[1]}/{report['channels']['analyzed']}**\n")
        f.write(f"- Global alignment shift: **{report['processing']['alignment_shift_ms']:.3f} ms**\n")
        f.write(f"- Mono gain match: **{g_mono:.6f}**\n")
        f.write(f"- Workers: **{max_workers}**\n\n")

        f.write("## MONO Metrics\n")
        f.write(f"- SDR: **{SDR_mono:.2f} dB**\n")
        if dominant_m:
            f.write(f"- Fundamental: **{f0_m:.2f} Hz**\n")
            f.write(f"- THD: **{100.0*thd_m:.4f} %**\n")
            f.write(f"- SINAD: **{sinad_m:.2f} dB**\n")
            f.write(f"- ENOB: **{enob_m:.3f} bits**\n")
        else:
            f.write("- THD / SINAD / ENOB: N/A (no dominant single-tone detected; use --f0 if needed)\n")

        f.write("\n## Per-Channel\n")
        for ch in range(report['channels']['analyzed']):
            chrep = results[ch]
            f.write(f"\n### Channel {ch}\n")
            f.write(f"- Gain match: **{chrep['gain_match']:.6f}**\n")
            f.write(f"- SDR: **{chrep['SDR_dB']:.2f} dB**\n")
            if chrep['dominant_tone_detected']:
                f.write(f"- Fundamental: **{chrep['fundamental_hz']:.2f} Hz**\n")
                f.write(f"- THD: **{chrep['THD_percent']:.4f} %**\n")
                f.write(f"- SINAD: **{chrep['SINAD_dB']:.2f} dB**\n")
                f.write(f"- ENOB: **{chrep['ENOB_bits']:.3f} bits**\n")
            else:
                f.write("- THD / SINAD / ENOB: N/A (no dominant single-tone detected; use --f0 if needed)\n")

            base_rel = f"ch{ch:02d}"
            f.write(f"![Before Spectrum]({base_rel}/spectrum_before.png)\n\n")
            f.write(f"![After Spectrum]({base_rel}/spectrum_after.png)\n\n")
            f.write(f"![Residual Spectrum]({base_rel}/spectrum_residual.png)\n\n")

        f.write("\n*Generated by audio_distortion_report_mc.py*\n")

    print(f"✅ Multi-channel report written to: {os.path.abspath(args.out)}")
    print(f"   - JSON : {json_path}")
    print(f"   - Markdown : {md_path}")
    print(f"   - Plots folder per channel: chXX/*.png, plus mono_*.png")

if __name__ == "__main__":
    main()