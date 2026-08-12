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

OUTPUT_XLSX_NAME = "flow_cytometry_analysis.xlsx"

# ------------------------- Gaussian fitting core (unchanged) -------------------------
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

def fit_ploidy_peaks(values, min_channel=0.0, bins=300, n_peaks=3, n_restarts=50, seed=42,
                      max_plausible_cv=20.0, scale_to_1024=True, raw_max=32768, preview_mode=False):
    """
    Fit exactly `n_peaks` Gaussians (user‑defined). Scaling to 1024.
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
    n_data = len(y_fit)

    if max_hist <= 0:
        result["note"] = "No signal in histogram."
        return result

    rng = np.random.default_rng(seed)

    # Use fewer restarts for preview
    actual_restarts = 10 if preview_mode else n_restarts

    best = _fit_fixed_n(
        x_fit, y_fit, n_peaks, values.min(), values.max(),
        bin_width, max_hist, actual_restarts, rng
    )
    if best is None:
        result["note"] = "Peak fitting failed."
        return result

    sse, peaks, curve, bg_amp, bg_k = best
    merged_peaks = _merge_close_peaks(peaks)
    # Quality filter: drop peaks with CV > max_plausible_cv
    merged_peaks = [p for p in merged_peaks if p["cv_percent"] is not None and p["cv_percent"] <= max_plausible_cv]

    result["peaks"] = merged_peaks
    result["fit_curve"] = (x_fit, curve)
    result["success"] = bool(merged_peaks)
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
                      standard_tolerance_percent=20.0,
                      min_channel=0.0,
                      n_peaks=3, n_restarts=50, max_plausible_cv=20.0,
                      scale_to_1024=True, raw_max=32768):
    try:
        if not uploaded_files:
            return "No files uploaded.", pd.DataFrame()

        batch_summary_rows = []
        parsed_cache = {}
        fit_cache = {}

        # First pass: parse and fit all files with user-defined n_peaks
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
                        n_peaks=n_peaks,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
                        scale_to_1024=scale_to_1024,
                        raw_max=raw_max,
                        preview_mode=False
                    )
            except Exception:
                pass

        # Get standard reference from standard file
        standard_reference_mean = None
        if standard_filename and standard_filename in fit_cache:
            std_peaks = fit_cache[standard_filename].get("peaks", [])
            if std_peaks:
                standard_reference_mean = std_peaks[0]["mu"]

        today_date = datetime.now().strftime("%Y-%m-%d")

        # Second pass: build rows
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                if filename not in parsed_cache:
                    raise ValueError("File failed to parse.")
                meta, data = parsed_cache[filename]
                numeric_data = data.select_dtypes(include="number")

                fit = fit_cache.get(filename)
                peaks_data = {}
                embryo_standard = endosperm_standard = endosperm_embryo = ""

                if fit and fit["success"]:
                    all_peaks = fit["peaks"]  # sorted by mu

                    # Identify Standard peak by closeness to reference
                    std_peak = None
                    std_idx = -1
                    if standard_reference_mean is not None:
                        tol = standard_tolerance_percent / 100.0
                        best_diff = np.inf
                        for idx, p in enumerate(all_peaks):
                            diff = abs(p["mu"] - standard_reference_mean) / standard_reference_mean
                            if diff < best_diff:
                                best_diff = diff
                                std_peak = p
                                std_idx = idx
                        if best_diff > tol:
                            std_peak = None
                            std_idx = -1

                    # Remove standard from list, label remaining by order
                    non_std = [p for i, p in enumerate(all_peaks) if i != std_idx]
                    non_std.sort(key=lambda p: p["mu"])

                    # Labeling: first non‑std = Embryo, second = Endosperm, rest = Extra_*
                    labels = ["Embryo", "Endosperm", "Embryo_G2", "Endosperm_G2"]
                    extra_counter = 1
                    for i, p in enumerate(non_std):
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

                    # Ratios
                    embryo_mean = peaks_data.get("Embryo", {}).get("mean", "")
                    endosperm_mean = peaks_data.get("Endosperm", {}).get("mean", "")
                    standard_mean = peaks_data.get("Standard", {}).get("mean", "")

                    if embryo_mean != "" and standard_mean != "":
                        embryo_standard = round(float(embryo_mean / standard_mean), 4)
                    if endosperm_mean != "" and standard_mean != "":
                        endosperm_standard = round(float(endosperm_mean / standard_mean), 4)
                    if embryo_mean != "" and endosperm_mean != "":
                        endosperm_embryo = round(float(endosperm_mean / embryo_mean), 4)

                row = {"File": filename, "date_FCM": today_date}

                # Always include Standard, Embryo, Endosperm, and any extras
                peak_order = ["Standard", "Embryo", "Endosperm"]
                # Add G2 and extras if present
                extra_keys = sorted([k for k in peaks_data.keys() if k not in peak_order])
                peak_order.extend(extra_keys)

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
                batch_summary_rows.append({"File": filename, "date_FCM": today_date})

        df = pd.DataFrame(batch_summary_rows)
        cols = [c for c in df.columns if c not in ["File", "date_FCM"]]
        cols = ["File"] + sorted(cols) + ["date_FCM"]
        df = df[cols]

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Analysis", index=False)
        excel_bytes = excel_buffer.getvalue()

        summary_text = f"Batch complete. {len(df)} files processed."
        return summary_text, df, excel_bytes

    except Exception:
        error_trace = traceback.format_exc()
        return f"Error:\n{error_trace}", pd.DataFrame(), None

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Flow Cytometry Analysis", layout="wide")
st.title("Flow Cytometry Analysis")
st.write(
    "Upload .fcs files and set the number of peaks to fit. "
    "Data is scaled to 1024 channels – standard should appear around ~100."
)

uploaded_files = st.file_uploader("Upload .fcs Files", type=["fcs"], accept_multiple_files=True)

dna_channel = None
standard_filename = None
n_peaks = 3
n_restarts = 50
min_channel = 0.0
standard_tolerance_percent = 20.0
max_plausible_cv = 20.0
raw_max = 32768

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

    with st.expander("Advanced settings"):
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            n_peaks = st.number_input("Number of peaks to fit", min_value=1, max_value=8, value=3, step=1,
                help="Set to the number of expected peaks (e.g., 3 for Standard, Embryo, Endosperm).")
        with col4:
            n_restarts = st.number_input("Fit restarts", min_value=10, max_value=500, value=50, step=10)
        with col5:
            min_channel = st.number_input("Debris cutoff", min_value=0.0, value=0.0, step=10.0)
        with col6:
            standard_tolerance_percent = st.number_input("Std tolerance (%)", min_value=5.0, max_value=100.0, value=20.0, step=5.0)

        col7, col8 = st.columns(2)
        with col7:
            max_plausible_cv = st.number_input("Max plausible CV%", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        with col8:
            scale_checked = st.checkbox("Scale to 1024", value=True)
        raw_max_override = st.number_input("Override raw max", min_value=1, value=raw_max, step=1)

    # Preview
    st.subheader("Preview fit")
    preview_file = st.selectbox("File to preview", options=filenames, key="preview")
    if st.button("Show Preview") and dna_channel:
        with st.spinner("Fitting (fast preview mode)..."):
            pfile = next(f for f in uploaded_files if f.name == preview_file)
            try:
                _, pdata = _parse_fcs(pfile)
                pdata = pdata.dropna()
                pnumeric = pdata.select_dtypes(include="number")
                if dna_channel in pnumeric.columns:
                    fit = fit_ploidy_peaks(
                        pnumeric[dna_channel].values,
                        min_channel=min_channel,
                        n_peaks=n_peaks,
                        n_restarts=n_restarts,
                        max_plausible_cv=max_plausible_cv,
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
                            st.write(f"Peak {i+1}: mean={p['mu']:.1f}, CV={p['cv_percent']:.2f}%")
                    else:
                        st.warning("No peaks fitted.")
                    st.caption(f"Scaling: raw / {raw_max_override} * 1023")
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
        with st.spinner("Processing..."):
            summary, df, excel = analyze_fcs_batch(
                uploaded_files, dna_channel, standard_filename,
                standard_tolerance_percent,
                min_channel,
                n_peaks, n_restarts, max_plausible_cv,
                scale_to_1024=scale_checked,
                raw_max=raw_max_override
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
