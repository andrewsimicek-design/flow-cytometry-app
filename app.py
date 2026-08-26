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

def fit_ploidy_peaks(values, min_channel=0.0, bins=300, n_peaks=8, n_restarts=100, seed=42,
                      max_plausible_cv=15.0, min_peak_height=0.0, min_amplitude_ratio=5.0,
                      scale_to_1024=True, raw_max=32768, preview_mode=False):
    """
    Fit peaks with smart filtering to detect only real peaks.
    - min_amplitude_ratio: peak must be at least this % of the main peak
    - min_peak_height: absolute minimum count
    - max_plausible_cv: CV threshold for real peaks
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

    candidates = {}
    for n_peaks_attempt in range(1, n_peaks + 1):
        best = _fit_fixed_n(
            x_fit, y_fit, n_peaks_attempt, values.min(), values.max(),
            bin_width, max_hist, actual_restarts, rng
        )
        if best is not None:
            sse, peaks, curve, bg_amp, bg_k = best
            merged_peaks = _merge_close_peaks(peaks, min_separation_sigma=2.0)
            
            # Smart filtering
            filtered = []
            if merged_peaks:
                max_amp = max(p["amp"] for p in merged_peaks)
                for p in merged_peaks:
                    # Condition 1: CV must be reasonable
                    if p["cv_percent"] is not None and p["cv_percent"] > max_plausible_cv:
                        continue
                    # Condition 2: amplitude must be > min_amplitude_ratio % of main peak
                    if p["amp"] < (min_amplitude_ratio / 100.0) * max_amp:
                        continue
                    # Condition 3: absolute minimum height
                    if p["amp"] < min_peak_height:
                        continue
                    filtered.append(p)
            
            candidates[n_peaks_attempt] = {
                "sse": sse, "peaks": filtered, "curve": curve,
                "bg_amp": bg_amp, "bg_k": bg_k,
            }

    if not candidates:
        result["note"] = "No peaks could be fit."
        return result

    n_data = len(y_fit)
    def bic(sse, n_peaks_attempt):
        sse = max(sse, 1e-9)
        k = 3 * n_peaks_attempt + 2
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
                      n_peaks=8, n_restarts=100, max_plausible_cv=15.0,
                      min_peak_height=0.0, min_amplitude_ratio=5.0,
                      scale_to_1024=True, raw_max=32768,
                      manual_standard_mean=None,
                      force_lowest_peak_as_standard=True):
    try:
        if not uploaded_files:
            return "No files uploaded.", pd.DataFrame()

        batch_summary_rows = []
        parsed_cache = {}
        fit_cache = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Processing {filename}...")
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
                        n_peaks=n_peaks,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
                        min_peak_height=min_peak_height,
                        min_amplitude_ratio=min_amplitude_ratio,
                        scale_to_1024=scale_to_1024,
                        raw_max=raw_max,
                        preview_mode=False
                    )
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(uploaded_files))

        progress_bar.empty()
        status_text.empty()

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

        summary_text = f"✅ Batch complete. {len(df)} files processed."
        return summary_text, df, excel_bytes

    except Exception as e:
        error_trace = traceback.format_exc()
        return f"❌ Error:\n{error_trace}", pd.DataFrame(), None

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Flow Cytometry Analysis", layout="wide", page_icon="🧬")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    st.subheader("📁 Files")
    uploaded_files = st.file_uploader("Upload .fcs Files", type=["fcs"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("🎯 Peak Detection")
    dna_channel = None
    standard_filename = None

    if uploaded_files:
        channels, filenames, raw_max = peek_channels_and_files(uploaded_files)
        if channels:
            dna_channel = st.selectbox("🧬 DNA Channel", options=channels)
        standard_filename = st.selectbox("📌 Standard File", options=["None"] + filenames)
        if standard_filename == "None":
            standard_filename = None

    st.markdown("---")
    st.subheader("🔬 Advanced")

    max_peaks = st.slider("Max peaks to detect", 1, 12, 8, help="Higher = catches small peaks")
    n_restarts = st.slider("Fit restarts", 10, 500, 100, help="Higher = more accurate, slower")
    min_channel = st.number_input("Debris cutoff", min_value=0.0, value=0.0, step=10.0)
    max_plausible_cv = st.slider("Max plausible CV%", 1.0, 30.0, 15.0, 1.0, help="Reject peaks above this")
    min_peak_height = st.number_input("Min peak height (counts)", min_value=0.0, value=0.0, step=1.0, help="0 = detect everything")
    min_amplitude_ratio = st.slider("Min amplitude ratio (% of main)", 1, 20, 5, help="Peaks smaller than this % of main peak are rejected")
    standard_tolerance_percent = st.slider("Standard tolerance %", 5, 100, 30, 5)

    st.markdown("---")
    st.subheader("📐 Scaling")
    scale_checked = st.checkbox("Scale to 1024 channels", value=True)
    raw_max_override = st.number_input("Override raw max", min_value=1, value=32768, step=1000)

    st.markdown("---")
    st.subheader("🏷️ Labels")
    force_lowest_peak = st.checkbox("Force lowest peak as Standard", value=True)
    manual_standard_mean = st.number_input("Manual Standard Mean (0 = auto)", min_value=0.0, value=0.0, step=1.0)
    if manual_standard_mean == 0:
        manual_standard_mean = None

# Main content
st.title("🧬 Flow Cytometry Analysis")
st.caption("Upload FCS files, adjust sensitivity, and export clean data.")

if not uploaded_files:
    st.info("📂 Please upload .fcs files in the sidebar to begin.")
    st.stop()

# Preview section
st.subheader("🔍 Preview a File")
preview_file = st.selectbox("Select file to preview", [f.name for f in uploaded_files])

if st.button("📊 Show Preview"):
    if not dna_channel:
        st.warning("Please select a DNA channel in the sidebar.")
    else:
        pfile = next(f for f in uploaded_files if f.name == preview_file)
        with st.spinner("Fitting..."):
            try:
                _, pdata = _parse_fcs(pfile)
                pdata = pdata.dropna()
                pnumeric = pdata.select_dtypes(include="number")
                if dna_channel in pnumeric.columns:
                    fit = fit_ploidy_peaks(
                        pnumeric[dna_channel].values,
                        min_channel=min_channel,
                        n_peaks=max_peaks,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
                        min_peak_height=min_peak_height,
                        min_amplitude_ratio=min_amplitude_ratio,
                        scale_to_1024=scale_checked,
                        raw_max=raw_max_override,
                        preview_mode=True
                    )

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        if fit["hist"] is not None:
                            bc, h = fit["hist"]
                            ax.bar(bc, h, width=(bc[1]-bc[0]) if len(bc)>1 else 1, alpha=0.5, color="steelblue")
                        if fit["fit_curve"] is not None:
                            xf, yf = fit["fit_curve"]
                            ax.plot(xf, yf, color="red", linewidth=2, label="Gaussian fit")
                        ax.set_xlabel(f"{dna_channel} (scaled)")
                        ax.set_ylabel("Count")
                        ax.legend()
                        st.pyplot(fig)

                    with col2:
                        st.subheader("Detected Peaks")
                        if fit["success"] and fit["peaks"]:
                            peak_data = []
                            for i, p in enumerate(fit["peaks"]):
                                peak_data.append({
                                    "Peak": i+1,
                                    "Mean": f"{p['mu']:.1f}",
                                    "CV%": f"{p['cv_percent']:.2f}",
                                    "Height": f"{p['amp']:.1f}",
                                    "% of main": f"{(p['amp'] / max([pp['amp'] for pp in fit['peaks']]) * 100):.1f}%"
                                })
                            st.dataframe(pd.DataFrame(peak_data), use_container_width=True)
                        else:
                            st.warning("No peaks detected.")
                else:
                    st.error(f"Channel '{dna_channel}' not found.")
            except Exception as e:
                st.error(f"Preview error: {e}")

# Run analysis
st.markdown("---")
col_run, col_reset = st.columns([3, 1])
with col_run:
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not dna_channel:
            st.warning("Please select a DNA channel in the sidebar.")
        elif not standard_filename:
            st.warning("Please select a standard file in the sidebar.")
        else:
            with st.spinner("Processing files..."):
                summary, df, excel = analyze_fcs_batch(
                    uploaded_files, dna_channel, standard_filename,
                    standard_tolerance_percent,
                    min_channel,
                    max_peaks, n_restarts, max_plausible_cv,
                    min_peak_height, min_amplitude_ratio,
                    scale_to_1024=scale_checked,
                    raw_max=raw_max_override,
                    manual_standard_mean=manual_standard_mean,
                    force_lowest_peak_as_standard=force_lowest_peak
                )
            st.session_state.batch_results = (summary, df, excel)

with col_reset:
    if st.button("🗑️ Clear Results", use_container_width=True):
        if "batch_results" in st.session_state:
            del st.session_state.batch_results
        st.rerun()

# Show results
if "batch_results" in st.session_state:
    results = st.session_state.batch_results
    if results is not None and len(results) == 3:
        summary, df, excel = results
        st.success(summary)

        st.subheader("📊 Results Preview")
        st.dataframe(df, use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Download Excel",
                data=excel,
                file_name=OUTPUT_XLSX_NAME,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with col_dl2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=OUTPUT_XLSX_NAME.replace(".xlsx", ".csv"),
                mime="text/csv",
                use_container_width=True
            )
