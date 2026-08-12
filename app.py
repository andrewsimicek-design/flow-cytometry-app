import os
import io
import tempfile
import traceback
from datetime import datetime

import numpy as np
import streamlit as st
import pandas as pd
import fcsparser
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

OUTPUT_XLSX_NAME = "flow_cytometry_analysis.xlsx"


# ------------------------- Gaussian fitting core (unchanged) -------------------------
def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _multi_gaussian(x, *params):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp, mu, sigma = params[i:i + 3]
        y += _gaussian(x, amp, mu, sigma)
    return y


def _make_model_with_background(x_min):
    def _model(x, bg_amp, bg_k, *gauss_params):
        bg = bg_amp * np.exp(-bg_k * (x - x_min))
        return bg + _multi_gaussian(x, *gauss_params)
    return _model


def _fit_fixed_n(x_fit, y_fit, n_peaks, values_min, values_max, bin_width, max_hist, n_restarts, rng):
    span = max(values_max - values_min, 1e-6)
    sigma_floor = max(bin_width * 1.2, 1e-6)
    model_func = _make_model_with_background(values_min)
    best = None

    for _ in range(n_restarts):
        bg_amp0 = rng.uniform(0, max_hist * 0.6)
        bg_k0 = rng.uniform(0, 15.0 / span)

        mus0 = np.sort(rng.uniform(values_min, values_max, n_peaks))
        sigmas0 = rng.uniform(max(sigma_floor, span * 0.003), span * 0.3, n_peaks)
        amps0 = rng.uniform(0.05 * max_hist, max(max_hist * 1.2, 1.0), n_peaks)

        p0 = [bg_amp0, bg_k0]
        lower = [0, 0]
        upper = [max_hist, 200.0 / span]
        for i in range(n_peaks):
            p0.extend([amps0[i], mus0[i], sigmas0[i]])
            lower.extend([0, values_min, sigma_floor])
            upper.extend([np.inf, values_max, span])

        try:
            popt, _ = curve_fit(model_func, x_fit, y_fit, p0=p0, bounds=(lower, upper), maxfev=4000)
        except Exception:
            continue

        model = model_func(x_fit, *popt)
        sse = float(np.sum((model - y_fit) ** 2))

        if best is None or sse < best[0]:
            bg_amp, bg_k = popt[0], popt[1]
            gauss_params = popt[2:]
            peaks = []
            for i in range(0, len(gauss_params), 3):
                amp, mu, sigma = gauss_params[i:i + 3]
                cv = abs(sigma / mu) * 100 if mu != 0 else None
                peaks.append({"mu": float(mu), "sigma": float(sigma), "amp": float(amp), "cv_percent": cv})
            peaks.sort(key=lambda p: p["mu"])
            best = (sse, peaks, model, float(bg_amp), float(bg_k))

    return best


def _merge_close_peaks(peaks, min_separation_sigma=1.0):
    if len(peaks) < 2:
        return peaks

    peaks = sorted(peaks, key=lambda p: p["mu"])
    merged = [peaks[0]]

    for p in peaks[1:]:
        last = merged[-1]
        avg_sigma = (last["sigma"] + p["sigma"]) / 2.0
        if avg_sigma > 0 and abs(p["mu"] - last["mu"]) < min_separation_sigma * avg_sigma:
            w1 = last["amp"] * last["sigma"]
            w2 = p["amp"] * p["sigma"]
            total_w = w1 + w2
            if total_w <= 0:
                continue
            combined_mu = (w1 * last["mu"] + w2 * p["mu"]) / total_w
            combined_var = (
                w1 * (last["sigma"] ** 2 + (last["mu"] - combined_mu) ** 2)
                + w2 * (p["sigma"] ** 2 + (p["mu"] - combined_mu) ** 2)
            ) / total_w
            combined_sigma = float(np.sqrt(max(combined_var, 1e-12)))
            combined_area = w1 * np.sqrt(2 * np.pi) + w2 * np.sqrt(2 * np.pi)
            combined_amp = combined_area / (combined_sigma * np.sqrt(2 * np.pi))
            cv = abs(combined_sigma / combined_mu) * 100 if combined_mu != 0 else None
            merged[-1] = {
                "mu": combined_mu, "sigma": combined_sigma,
                "amp": combined_amp, "cv_percent": cv,
            }
        else:
            merged.append(p)

    return merged


def _estimate_peak_count(hist_values, min_distance=5, prominence_rel=0.02):
    peak_indices, properties = find_peaks(
        hist_values,
        prominence=np.ptp(hist_values) * prominence_rel,
        distance=min_distance
    )
    return min(len(peak_indices), 8)


def fit_ploidy_peaks(values, min_channel=0.0, bins=300, n_restarts=50, seed=42,
                      max_plausible_cv=20.0, scale_to_1024=True, raw_max=32768,
                      auto_detect_peaks=True, max_peaks=8, preview_mode=False):
    """
    Fit Gaussian peaks to the data. preview_mode speeds up the fit for previews.
    """
    values = np.asarray(values, dtype=float)
    
    if scale_to_1024 and raw_max > 0:
        values = values / raw_max * 1023
    
    values = values[values >= min_channel]

    result = {"success": False, "peaks": [], "hist": None, "fit_curve": None, "note": ""}

    if values.size < 20:
        result["note"] = "Too few events after debris cutoff to fit peaks."
        return result

    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    result["hist"] = (bin_centers, hist)
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    x_fit = bin_centers
    y_fit = hist.astype(float)
    max_hist = float(hist.max())
    n_data = len(y_fit)

    if max_hist <= 0:
        result["note"] = "No signal in histogram."
        return result

    rng = np.random.default_rng(seed)

    # ---- AUTO-DETECT PEAK COUNT ----
    if auto_detect_peaks:
        suggested_n = _estimate_peak_count(y_fit)
        max_peaks_to_try = min(suggested_n + 1, max_peaks)
        if max_peaks_to_try < 2 and suggested_n < 2:
            max_peaks_to_try = 3
        peak_counts_to_try = set()
        # In preview mode, only try the suggested number and one each side
        if preview_mode:
            for n in range(max(1, suggested_n - 1), min(max_peaks, suggested_n + 2)):
                peak_counts_to_try.add(n)
            # Always include 1, 2, 3 for safety
            for n in [1, 2, 3]:
                if n <= max_peaks:
                    peak_counts_to_try.add(n)
        else:
            for n in range(max(1, suggested_n - 1), max_peaks_to_try + 1):
                peak_counts_to_try.add(n)
            for n in [1, 2, 3]:
                if n <= max_peaks:
                    peak_counts_to_try.add(n)
        peak_counts_to_try = sorted(peak_counts_to_try)
    else:
        peak_counts_to_try = list(range(1, max_peaks + 1))

    # Reduce restarts for preview mode
    actual_restarts = 10 if preview_mode else n_restarts

    candidates = {}
    for n_peaks in peak_counts_to_try:
        if n_peaks > max_peaks:
            continue
        best = _fit_fixed_n(
            x_fit, y_fit, n_peaks, values.min(), values.max(), bin_width, max_hist, actual_restarts, rng
        )
        if best is not None:
            sse, peaks, curve, bg_amp, bg_k = best
            merged_peaks = _merge_close_peaks(peaks)
            is_plausible = all(
                p["cv_percent"] is not None and p["cv_percent"] <= max_plausible_cv
                for p in merged_peaks
            )
            candidates[n_peaks] = {
                "sse": sse, "peaks": merged_peaks, "curve": curve,
                "bg_amp": bg_amp, "bg_k": bg_k, "plausible": is_plausible,
            }

    if not candidates:
        result["note"] = "No peaks could be fit."
        return result

    def bic(sse, n_peaks):
        sse = max(sse, 1e-9)
        k = 3 * n_peaks + 2
        return n_data * np.log(sse / n_data) + k * np.log(n_data)

    plausible_ns = [n for n, c in candidates.items() if c["plausible"]]
    pool = plausible_ns if plausible_ns else list(candidates.keys())

    best_n = min(pool, key=lambda n: bic(candidates[n]["sse"], n))
    # Prefer simpler plausible models
    for n in sorted(pool):
        if candidates[n]["plausible"] and n < best_n:
            bic_diff = bic(candidates[n]["sse"], n) - bic(candidates[best_n]["sse"], best_n)
            if bic_diff < 5:
                best_n = n
                break

    chosen = candidates[best_n]
    sse, peaks, curve, bg_amp, bg_k = chosen["sse"], chosen["peaks"], chosen["curve"], chosen["bg_amp"], chosen["bg_k"]

    result["peaks"] = peaks
    result["fit_curve"] = (x_fit, curve)
    result["success"] = True
    result["note"] = ""
    return result


# ------------------------- FCS parsing helpers -------------------------
def _parse_fcs(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fcs") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        meta, data = fcsparser.parse(tmp_path, reformat_meta=True)
    finally:
        os.remove(tmp_path)
    return meta, data


def peek_channels_and_files(uploaded_files):
    channels = []
    raw_max = 32768
    if uploaded_files:
        try:
            meta, data = _parse_fcs(uploaded_files[0])
            data = data.dropna()
            channels = list(data.select_dtypes(include="number").columns)
            if meta and '$P1R' in meta:
                raw_max = int(meta['$P1R'])
            elif meta and '$PnR' in meta:
                raw_max = int(meta['$PnR'])
        except Exception:
            pass
    filenames = [f.name for f in uploaded_files] if uploaded_files else []
    return channels, filenames, raw_max


# ------------------------- Core batch analysis (unchanged) -------------------------
def analyze_fcs_batch(uploaded_files, dna_channel,
                      standard_filename,
                      standard_tolerance_percent=20.0,
                      min_channel=0.0,
                      n_restarts=50, max_plausible_cv=20.0,
                      scale_to_1024=True, raw_max=32768,
                      auto_detect_peaks=True, max_peaks=8):
    try:
        if not uploaded_files:
            return "No files were uploaded.", pd.DataFrame()

        batch_summary_rows = []

        parsed_cache = {}
        fit_cache = {}

        # First pass: parse and fit all files
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                meta, data = _parse_fcs(uploaded_file)
                data = data.dropna()
                if data.empty:
                    raise ValueError("No usable event data remained after cleaning.")
                parsed_cache[filename] = (meta, data)

                numeric_data = data.select_dtypes(include="number")
                if dna_channel in numeric_data.columns:
                    fit_cache[filename] = fit_ploidy_peaks(
                        numeric_data[dna_channel].values,
                        min_channel=min_channel,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
                        scale_to_1024=scale_to_1024,
                        raw_max=raw_max,
                        auto_detect_peaks=auto_detect_peaks,
                        max_peaks=max_peaks,
                        preview_mode=False
                    )
            except Exception:
                pass

        standard_reference_mean = None
        if standard_filename and standard_filename in fit_cache:
            std_peaks = fit_cache[standard_filename].get("peaks", [])
            if std_peaks:
                standard_reference_mean = std_peaks[0]["mu"]

        today_date = datetime.now().strftime("%Y-%m-%d")

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                if filename not in parsed_cache:
                    raise ValueError("File failed to parse.")
                meta, data = parsed_cache[filename]
                numeric_data = data.select_dtypes(include="number")

                peaks_data = {}
                embryo_standard = endosperm_standard = endosperm_embryo = ""

                fit = fit_cache.get(filename)
                if fit and fit["success"]:
                    all_peaks = fit["peaks"]

                    std_peak = None
                    std_peak_idx = -1
                    if standard_reference_mean is not None:
                        tol = standard_tolerance_percent / 100.0
                        best_diff = np.inf
                        for idx, p in enumerate(all_peaks):
                            diff = abs(p["mu"] - standard_reference_mean) / standard_reference_mean
                            if diff < best_diff:
                                best_diff = diff
                                std_peak = p
                                std_peak_idx = idx
                        if best_diff > tol:
                            std_peak = None
                            std_peak_idx = -1

                    non_std_peaks = [p for i, p in enumerate(all_peaks) if i != std_peak_idx]
                    non_std_peaks.sort(key=lambda p: p["mu"])

                    labels = ["Embryo", "Endosperm", "Embryo_G2", "Endosperm_G2"]
                    extra_counter = 1

                    for i, p in enumerate(non_std_peaks):
                        if i < len(labels):
                            label = labels[i]
                        else:
                            label = f"Extra_{extra_counter}"
                            extra_counter += 1
                        peaks_data[label] = {
                            "mean": round(float(p["mu"]), 3),
                            "cv": round(p["cv_percent"], 3) if p["cv_percent"] is not None else ""
                        }

                    if std_peak:
                        peaks_data["Standard"] = {
                            "mean": round(float(std_peak["mu"]), 3),
                            "cv": round(std_peak["cv_percent"], 3) if std_peak["cv_percent"] is not None else ""
                        }

                    embryo_mean = peaks_data.get("Embryo", {}).get("mean", "")
                    endosperm_mean = peaks_data.get("Endosperm", {}).get("mean", "")
                    standard_mean = peaks_data.get("Standard", {}).get("mean", "")

                    if embryo_mean != "" and standard_mean != "":
                        embryo_standard = round(float(embryo_mean / standard_mean), 4)
                    if endosperm_mean != "" and standard_mean != "":
                        endosperm_standard = round(float(endosperm_mean / standard_mean), 4)
                    if embryo_mean != "" and endosperm_mean != "":
                        endosperm_embryo = round(float(endosperm_mean / embryo_mean), 4)

                row = {
                    "File": filename,
                    "date_FCM": today_date,
                }

                peak_order = ["Standard", "Embryo", "Endosperm", "Embryo_G2", "Endosperm_G2"]
                extra_labels = sorted([k for k in peaks_data.keys() if k.startswith("Extra_")])
                peak_order.extend(extra_labels)

                for label in peak_order:
                    if label in peaks_data:
                        row[f"{label} Mean"] = peaks_data[label]["mean"]
                        row[f"{label} CV"] = peaks_data[label]["cv"]
                    else:
                        row[f"{label} Mean"] = ""
                        row[f"{label} CV"] = ""

                row["Embryo/Standard"] = embryo_standard
                row["Endosperm/Standard"] = endosperm_standard
                row["Endosperm/Embryo"] = endosperm_embryo

                batch_summary_rows.append(row)

            except Exception:
                batch_summary_rows.append({
                    "File": filename,
                    "date_FCM": today_date,
                })

        df = pd.DataFrame(batch_summary_rows)
        cols = [c for c in df.columns if c not in ["File", "date_FCM"]]
        cols = ["File"] + sorted(cols) + ["date_FCM"]
        df = df[cols]

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Analysis", index=False)
        excel_bytes = excel_buffer.getvalue()

        summary_text = f"Batch processing complete. {len(df)} files processed."
        return summary_text, df, excel_bytes

    except Exception:
        error_trace = traceback.format_exc()
        return f"Fatal error during batch processing:\n{error_trace}", pd.DataFrame(), None


# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Flow Cytometry Analysis", layout="wide")
st.title("Flow Cytometry Analysis")
st.write(
    "Upload your `.fcs` files and select the internal standard file.\n\n"
    "**Features:**\n"
    "- Automatically detects any number of peaks (not just 3).\n"
    "- Labels peaks as: Standard, Embryo, Endosperm, Embryo_G2, Endosperm_G2, Extra_1, ...\n"
    "- Scales to 1024-channel scale (standard at ~100).\n"
    "- Preview uses faster settings to avoid long waits."
)

uploaded_files = st.file_uploader(
    "Upload .fcs Files", type=["fcs"], accept_multiple_files=True,
)

dna_channel = None
standard_filename = None
standard_tolerance_percent = 20.0
min_channel = 0.0
max_peaks = 8
n_restarts = 50
max_plausible_cv = 20.0
raw_max = 32768

if uploaded_files:
    channels, filenames, raw_max = peek_channels_and_files(uploaded_files)

    col1, col2 = st.columns(2)
    with col1:
        if channels:
            dna_channel = st.selectbox("DNA / PI Channel", options=channels)
        else:
            st.warning("Could not read channel names from the first file.")
    with col2:
        standard_filename = st.selectbox("Internal Standard file", options=filenames)

    with st.expander("Advanced settings (optional)"):
        col3, col4, col5, col6, col7 = st.columns(5)
        with col3:
            standard_tolerance_percent = st.number_input("Standard tolerance (%)", min_value=5.0, max_value=100.0, value=20.0, step=5.0)
        with col4:
            min_channel = st.number_input("Debris cutoff", min_value=0.0, value=0.0, step=10.0,
                help="Exclude events below this channel value (on the 0-1023 scale).")
        with col5:
            max_peaks = st.number_input("Max peaks (safety limit)", min_value=1, max_value=12, value=8, step=1)
        with col6:
            n_restarts = st.number_input("Fit restarts", min_value=10, max_value=500, value=50, step=10)
        with col7:
            st.number_input("Raw max (detected)", value=raw_max, disabled=True)
        
        col8, col9 = st.columns(2)
        with col8:
            scale_checked = st.checkbox("Scale to 1024 channels", value=True)
        with col9:
            raw_max_override = st.number_input("Override raw max", min_value=1, value=raw_max, step=1)

        auto_detect = st.checkbox("Auto-detect peak count", value=True)
        max_plausible_cv = st.number_input("Max plausible CV%", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

    # Preview
    st.subheader("Preview fit (fast mode)")
    preview_file = st.selectbox("File to preview", options=filenames, key="preview")
    if st.button("Show Preview") and dna_channel:
        with st.spinner("Fitting peaks (fast preview mode)... may take a few seconds"):
            pfile = next(f for f in uploaded_files if f.name == preview_file)
            try:
                _, pdata = _parse_fcs(pfile)
                pdata = pdata.dropna()
                pnumeric = pdata.select_dtypes(include="number")
                if dna_channel in pnumeric.columns:
                    # Use preview_mode=True to reduce restarts and peak count attempts
                    fit = fit_ploidy_peaks(
                        pnumeric[dna_channel].values,
                        min_channel=min_channel,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
                        scale_to_1024=scale_checked,
                        raw_max=raw_max_override,
                        auto_detect_peaks=auto_detect,
                        max_peaks=min(max_peaks, 5),  # cap at 5 for speed
                        preview_mode=True
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    if fit["hist"] is not None:
                        bc, h = fit["hist"]
                        ax.bar(bc, h, width=(bc[1] - bc[0]) if len(bc) > 1 else 1, alpha=0.5)
                    if fit["fit_curve"] is not None:
                        xf, yf = fit["fit_curve"]
                        ax.plot(xf, yf, color="red", linewidth=2)
                    ax.set_xlabel(f"{dna_channel} (scaled to 0-1023)")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                    if fit["success"]:
                        for i, p in enumerate(fit["peaks"]):
                            st.write(f"Peak {i+1}: mean={p['mu']:.1f}, CV={p['cv_percent']:.2f}%")
                        st.caption(f"Detected {len(fit['peaks'])} peaks (preview mode used fewer restarts)")
                    else:
                        st.warning("No peaks fitted.")
                    st.caption(f"Scaling: raw values divided by {raw_max_override} × 1023")
            except Exception as e:
                st.error(f"Preview error: {e}")

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if st.button("Run Analysis", type="primary"):
    if not uploaded_files:
        st.warning("Please upload files.")
    elif not dna_channel:
        st.warning("Select a DNA channel.")
    elif not standard_filename:
        st.warning("Select an internal standard file.")
    else:
        with st.spinner("Processing all files (this may take a while)..."):
            summary, df, excel = analyze_fcs_batch(
                uploaded_files, dna_channel,
                standard_filename,
                standard_tolerance_percent,
                min_channel,
                n_restarts, max_plausible_cv,
                scale_to_1024=scale_checked,
                raw_max=raw_max_override,
                auto_detect_peaks=auto_detect,
                max_peaks=max_peaks
            )
        st.session_state.batch_results = (summary, df, excel)

if st.session_state.batch_results is not None:
    summary, df, excel = st.session_state.batch_results
    st.success(summary)
    st.dataframe(df)
    st.download_button(
        label="Download Excel Report",
        data=excel,
        file_name=OUTPUT_XLSX_NAME,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if st.button("Clear results"):
        st.session_state.batch_results = None
        st.rerun()
