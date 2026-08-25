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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

OUTPUT_XLSX_NAME = "flow_cytometry_analysis.xlsx"

# ------------------------- Gaussian fitting core -------------------------
def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _multi_gaussian(x, *params):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp, mu, sigma = params[i:i+3]
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
                amp, mu, sigma = gauss_params[i:i+3]
                cv = abs(sigma / mu) * 100 if mu != 0 else None
                peaks.append({"mu": float(mu), "sigma": float(sigma), "amp": float(amp), "cv_percent": cv})
            peaks.sort(key=lambda p: p["mu"])
            best = (sse, peaks, model, float(bg_amp), float(bg_k))

    return best

def _merge_close_peaks(peaks, min_separation_sigma=2.0):
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
                w1 * (last["sigma"]**2 + (last["mu"] - combined_mu)**2)
                + w2 * (p["sigma"]**2 + (p["mu"] - combined_mu)**2)
            ) / total_w
            combined_sigma = float(np.sqrt(max(combined_var, 1e-12)))
            combined_area = w1 * np.sqrt(2*np.pi) + w2 * np.sqrt(2*np.pi)
            combined_amp = combined_area / (combined_sigma * np.sqrt(2*np.pi))
            cv = abs(combined_sigma / combined_mu) * 100 if combined_mu != 0 else None
            merged[-1] = {
                "mu": combined_mu, "sigma": combined_sigma,
                "amp": combined_amp, "cv_percent": cv,
            }
        else:
            merged.append(p)
    return merged

def fit_ploidy_peaks(values, min_channel=0.0, bins=300, max_peaks=8, n_restarts=100, seed=42,
                      max_plausible_cv=15.0, min_peak_height=0.0, scale_to_1024=True, raw_max=32768, preview_mode=False):
    """
    Ultra‑sensitive fitting: no amplitude filter, only CV and user‑set height threshold.
    """
    values = np.asarray(values, dtype=float)
    if scale_to_1024 and raw_max > 0:
        values = values / raw_max * 1023
    values = values[values >= min_channel]

    result = {"success": False, "peaks": [], "hist": None, "fit_curve": None, "note": ""}

    if values.size < 20:
        result["note"] = "Too few events after debris cutoff."
        return result

    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    result["hist"] = (bin_centers, hist)
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    x_fit = bin_centers
    y_fit = hist.astype(float)
    max_hist = float(hist.max())

    if max_hist <= 0:
        result["note"] = "No signal in histogram."
        return result

    rng = np.random.default_rng(seed)
    actual_restarts = 10 if preview_mode else n_restarts

    # Try all peak counts from 1 to max_peaks, pick best by BIC
    candidates = {}
    for n_peaks in range(1, max_peaks + 1):
        best = _fit_fixed_n(
            x_fit, y_fit, n_peaks, values.min(), values.max(),
            bin_width, max_hist, actual_restarts, rng
        )
        if best is not None:
            sse, peaks, curve, bg_amp, bg_k = best
            merged_peaks = _merge_close_peaks(peaks, min_separation_sigma=2.0)
            # Keep peaks with CV <= max_plausible_cv and height >= min_peak_height
            filtered = []
            for p in merged_peaks:
                if p["cv_percent"] is not None and p["cv_percent"] <= max_plausible_cv:
                    if p["amp"] >= min_peak_height:
                        filtered.append(p)
            candidates[n_peaks] = {
                "sse": sse, "peaks": filtered, "curve": curve,
                "bg_amp": bg_amp, "bg_k": bg_k,
            }

    if not candidates:
        result["note"] = "No peaks could be fit."
        return result

    # BIC selection
    n_data = len(y_fit)
    def bic(sse, n_peaks):
        sse = max(sse, 1e-9)
        k = 3 * n_peaks + 2
        return n_data * np.log(sse / n_data) + k * np.log(n_data)

    best_n = min(candidates.keys(), key=lambda n: bic(candidates[n]["sse"], n))
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

# ------------------------- Core batch analysis -------------------------
def analyze_fcs_batch(uploaded_files, dna_channel,
                      standard_filename,
                      standard_tolerance_percent=30.0,
                      min_channel=0.0,
                      max_peaks=8, n_restarts=100, max_plausible_cv=15.0, min_peak_height=0.0,
                      scale_to_1024=True, raw_max=32768,
                      manual_standard_mean=None,
                      force_lowest_peak_as_standard=True):
    try:
        if not uploaded_files:
            return "No files uploaded.", pd.DataFrame()

        batch_summary_rows = []
        parsed_cache = {}
        fit_cache = {}

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                meta, data = _parse_fcs(uploaded_file)
                data = data.dropna()
                if data.empty:
                    raise ValueError("No usable data.")
                parsed_cache[filename] = (meta, data)

                numeric_data = data.select_dtypes(include="number")
                if dna_channel in numeric_data.columns:
                    fit_cache[filename] = fit_ploidy_peaks(
                        numeric_data[dna_channel].values,
                        min_channel=min_channel,
                        max_peaks=max_peaks,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
                        min_peak_height=min_peak_height,
                        scale_to_1024=scale_to_1024,
                        raw_max=raw_max,
                        preview_mode=False
                    )
            except Exception:
                pass

        standard_reference_mean = manual_standard_mean
        if standard_filename and standard_filename in fit_cache and standard_reference_mean is None:
            std_peaks = fit_cache[standard_filename].get("peaks", [])
            if std_peaks:
                standard_reference_mean = std_peaks[0]["mu"]

        if standard_reference_mean is None and standard_filename and standard_filename in fit_cache:
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

                fit = fit_cache.get(filename)

                standard_mean = standard_cv = ""
                embryo_mean = embryo_cv = ""
                endosperm_mean = endosperm_cv = ""
                embryo_standard = endosperm_standard = endosperm_embryo = ""
                notes = ""

                if fit and fit["success"]:
                    all_peaks = fit["peaks"]

                    std_peak = None
                    std_idx = -1

                    if force_lowest_peak_as_standard and all_peaks:
                        std_peak = all_peaks[0]
                        std_idx = 0
                    elif standard_reference_mean is not None and all_peaks:
                        tol = standard_tolerance_percent / 100.0
                        best_diff = np.inf
                        for idx, p in enumerate(all_peaks):
                            diff = abs(p["mu"] - standard_reference_mean) / standard_reference_mean
                            if diff < best_diff:
                                best_diff = diff
                                std_peak = p
                                std_idx = idx
                        if best_diff > tol and all_peaks:
                            std_peak = all_peaks[0]
                            std_idx = 0
                    elif all_peaks:
                        std_peak = all_peaks[0]
                        std_idx = 0

                    if std_peak is not None:
                        non_std = [p for i, p in enumerate(all_peaks) if i != std_idx]
                        non_std.sort(key=lambda p: p["mu"])

                        if std_peak:
                            standard_mean = round(float(std_peak["mu"]), 3)
                            standard_cv = round(std_peak["cv_percent"], 3) if std_peak["cv_percent"] is not None else ""

                        if len(non_std) >= 1:
                            embryo_mean = round(float(non_std[0]["mu"]), 3)
                            embryo_cv = round(non_std[0]["cv_percent"], 3) if non_std[0]["cv_percent"] is not None else ""

                        if len(non_std) >= 2:
                            endosperm_mean = round(float(non_std[1]["mu"]), 3)
                            endosperm_cv = round(non_std[1]["cv_percent"], 3) if non_std[1]["cv_percent"] is not None else ""

                        # ---- Notes ----
                        if embryo_mean == "" and endosperm_mean == "":
                            notes = "len standard"
                        elif embryo_mean != "" and endosperm_mean == "":
                            notes = f"standard + embryo (CV {embryo_cv}%)"
                        elif embryo_mean != "" and endosperm_mean != "":
                            notes = f"standard + embryo (CV {embryo_cv}%) + endosperm (CV {endosperm_cv}%)"

                        # ---- Ratios ----
                        if embryo_mean != "" and standard_mean != "":
                            embryo_standard = round(float(embryo_mean / standard_mean), 4)
                        if endosperm_mean != "" and standard_mean != "":
                            endosperm_standard = round(float(endosperm_mean / standard_mean), 4)
                        if embryo_mean != "" and endosperm_mean != "":
                            endosperm_embryo = round(float(endosperm_mean / embryo_mean), 4)

                row = {
                    "Sample_ID": filename.replace(".fcs", ""),
                    "File_name": filename,
                    "Mean_1peak_[standard]": standard_mean,
                    "Mean_2peak_[embryo]": embryo_mean,
                    "Mean_3peak_[endosperm]": endosperm_mean,
                    "CV_1peak_[standard]": standard_cv,
                    "CV_2peak_[embryo]": embryo_cv,
                    "CV_3peak_[endosperm]": endosperm_cv,
                    "Embryo:standard_ratio_[Mean_2peak:Mean_1peak]": embryo_standard,
                    "Endosperm:standard_ratio_[Mean_3peak:Mean_1peak]": endosperm_standard,
                    "Endosperm:embryo_ratio_[Mean_3peak:Mean_2peak]": endosperm_embryo,
                    "": "",
                    "Date_of_analyses": today_date,
                    "Poznámka": notes,
                }
                batch_summary_rows.append(row)

            except Exception:
                batch_summary_rows.append({
                    "Sample_ID": filename.replace(".fcs", ""),
                    "File_name": filename,
                    "Mean_1peak_[standard]": "",
                    "Mean_2peak_[embryo]": "",
                    "Mean_3peak_[endosperm]": "",
                    "CV_1peak_[standard]": "",
                    "CV_2peak_[embryo]": "",
                    "CV_3peak_[endosperm]": "",
                    "Embryo:standard_ratio_[Mean_2peak:Mean_1peak]": "",
                    "Endosperm:standard_ratio_[Mean_3peak:Mean_1peak]": "",
                    "Endosperm:embryo_ratio_[Mean_3peak:Mean_2peak]": "",
                    "": "",
                    "Date_of_analyses": today_date,
                    "Poznámka": "error",
                })

        docent_columns = [
            "Sample_ID",
            "File_name",
            "Mean_1peak_[standard]",
            "Mean_2peak_[embryo]",
            "Mean_3peak_[endosperm]",
            "CV_1peak_[standard]",
            "CV_2peak_[embryo]",
            "CV_3peak_[endosperm]",
            "Embryo:standard_ratio_[Mean_2peak:Mean_1peak]",
            "Endosperm:standard_ratio_[Mean_3peak:Mean_1peak]",
            "Endosperm:embryo_ratio_[Mean_3peak:Mean_2peak]",
            "",
            "Date_of_analyses",
            "Poznámka"
        ]

        df = pd.DataFrame(batch_summary_rows, columns=docent_columns)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Analysis", index=False)
        excel_bytes = excel_buffer.getvalue()

        summary_text = f"Batch complete. {len(df)} files processed."
        return summary_text, df, excel_bytes

    except Exception as e:
        error_trace = traceback.format_exc()
        return f"Error:\n{error_trace}", pd.DataFrame(), None

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Flow Cytometry Analysis", layout="wide")
st.title("Flow Cytometry Analysis")
st.write("Upload `.fcs` files and adjust sensitivity to detect even small peaks.")

uploaded_files = st.file_uploader("Upload .fcs Files", type=["fcs"], accept_multiple_files=True)

dna_channel = None
standard_filename = None
max_peaks = 8
n_restarts = 100
min_channel = 0.0
standard_tolerance_percent = 30.0
max_plausible_cv = 15.0
min_peak_height = 0.0
raw_max = 32768
manual_standard_mean = None
force_lowest_peak = True

if uploaded_files:
    channels, filenames, raw_max = peek_channels_and_files(uploaded_files)

    col1, col2 = st.columns(2)
    with col1:
        if channels:
            dna_channel = st.selectbox("DNA / PI Channel", options=channels)
        else:
            st.warning("No numeric channels found.")
    with col2:
        standard_filename = st.selectbox("Internal Standard file", options=filenames)

    with st.expander("Advanced settings (for small peaks)"):
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            max_peaks = st.number_input("Max peaks to detect", min_value=1, max_value=12, value=8, step=1,
                help="Set higher if you expect many small peaks.")
        with col4:
            n_restarts = st.number_input("Fit restarts", min_value=10, max_value=500, value=100, step=10)
        with col5:
            min_channel = st.number_input("Debris cutoff", min_value=0.0, value=0.0, step=10.0)
        with col6:
            standard_tolerance_percent = st.number_input("Standard tolerance (%)", min_value=5.0, max_value=100.0, value=30.0, step=5.0)

        col7, col8, col9, col10 = st.columns(4)
        with col7:
            max_plausible_cv = st.number_input("Max plausible CV%", min_value=1.0, max_value=30.0, value=15.0, step=1.0,
                help="Peaks with CV above this are rejected. For real peaks use 10‑15%.")
        with col8:
            min_peak_height = st.number_input("Min peak height (counts)", min_value=0.0, value=0.0, step=1.0,
                help="Ignore peaks smaller than this (0 = detect everything). Set higher to filter noise.")
        with col9:
            scale_checked = st.checkbox("Scale to 1024", value=True)
        with col10:
            raw_max_override = st.number_input("Override raw max", min_value=1, value=raw_max, step=1)

        force_lowest_peak = st.checkbox("Force lowest peak as Standard", value=True)

        manual_standard_mean = st.number_input(
            "Manual Standard Mean (optional)",
            min_value=0.0, value=0.0, step=1.0,
            help="Enter expected standard mean (scaled). Leave 0 for auto."
        )
        if manual_standard_mean == 0:
            manual_standard_mean = None

    # Preview
    st.subheader("Preview fit")
    preview_file = st.selectbox("File to preview", options=filenames, key="preview")
    if st.button("Show Preview") and dna_channel:
        with st.spinner("Fitting (preview mode)..."):
            pfile = next(f for f in uploaded_files if f.name == preview_file)
            try:
                _, pdata = _parse_fcs(pfile)
                pdata = pdata.dropna()
                pnumeric = pdata.select_dtypes(include="number")
                if dna_channel in pnumeric.columns:
                    fit = fit_ploidy_peaks(
                        pnumeric[dna_channel].values,
                        min_channel=min_channel,
                        max_peaks=max_peaks,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
                        min_peak_height=min_peak_height,
                        scale_to_1024=scale_checked,
                        raw_max=raw_max_override,
                        preview_mode=True
                    )
                    fig, ax = plt.subplots(figsize=(8,4))
                    if fit["hist"] is not None:
                        bc, h = fit["hist"]
                        ax.bar(bc, h, width=(bc[1]-bc[0]) if len(bc)>1 else 1, alpha=0.5)
                    if fit["fit_curve"] is not None:
                        xf, yf = fit["fit_curve"]
                        ax.plot(xf, yf, color="red", linewidth=2)
                    ax.set_xlabel(f"{dna_channel} (scaled)")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                    if fit["success"]:
                        for i, p in enumerate(fit["peaks"]):
                            st.write(f"Peak {i+1}: mean={p['mu']:.1f}, CV={p['cv_percent']:.2f}% (height={p['amp']:.1f})")
                    else:
                        st.warning("No peaks fitted.")
                    st.caption(f"Scaling: raw / {raw_max_override} * 1023")
                    st.caption(f"CV threshold: {max_plausible_cv}%, min height: {min_peak_height}")
            except Exception as e:
                st.error(f"Preview error: {e}")

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if st.button("Run Analysis", type="primary"):
    if not uploaded_files:
        st.warning("Upload files.")
    elif not dna_channel:
        st.warning("Select a DNA channel.")
    elif not standard_filename:
        st.warning("Select a standard file.")
    else:
        with st.spinner("Processing (sensitive mode)..."):
            summary, df, excel = analyze_fcs_batch(
                uploaded_files, dna_channel, standard_filename,
                standard_tolerance_percent,
                min_channel,
                max_peaks, n_restarts, max_plausible_cv, min_peak_height,
                scale_to_1024=scale_checked,
                raw_max=raw_max_override,
                manual_standard_mean=manual_standard_mean,
                force_lowest_peak_as_standard=force_lowest_peak
            )
        st.session_state.batch_results = (summary, df, excel)

if st.session_state.batch_results is not None:
    summary, df, excel = st.session_state.batch_results
    st.success(summary)
    st.dataframe(df)
    st.download_button(
        label="Download Excel",
        data=excel,
        file_name=OUTPUT_XLSX_NAME,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if st.button("Clear results"):
        st.session_state.batch_results = None
        st.rerun()
