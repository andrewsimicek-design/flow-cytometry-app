import os
import io
import tempfile
import traceback

import numpy as np
import streamlit as st
import pandas as pd
import fcsparser
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

OUTPUT_XLSX_NAME = "flow_cytometry_analysis.xlsx"

# Minimal output columns – exactly what a manual analysis would report
OUTPUT_COLUMNS = [
    "File",
    "Embryo Mean",
    "Embryo CV",
    "Endosperm Mean",
    "Endosperm CV",
    "Standard Mean",
    "Standard CV",
    "Embryo/Standard",
    "Endosperm/Standard",
    "Endosperm/Embryo",
    "date_FCM",
]


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


def fit_ploidy_peaks(values, min_channel=0.0, bins=300, max_peaks=3, n_restarts=50, seed=42,
                      max_plausible_cv=20.0):
    values = np.asarray(values, dtype=float)
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

    candidates = {}
    for n_peaks in range(1, max_peaks + 1):
        best = _fit_fixed_n(
            x_fit, y_fit, n_peaks, values.min(), values.max(), bin_width, max_hist, n_restarts, rng
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
    chosen = candidates[best_n]
    sse, peaks, curve, bg_amp, bg_k = chosen["sse"], chosen["peaks"], chosen["curve"], chosen["bg_amp"], chosen["bg_k"]

    result["peaks"] = peaks
    result["fit_curve"] = (x_fit, curve)
    result["success"] = True
    result["note"] = ""  # keep note empty to hide any algorithm messages
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
    if uploaded_files:
        try:
            _, data = _parse_fcs(uploaded_files[0])
            data = data.dropna()
            channels = list(data.select_dtypes(include="number").columns)
        except Exception:
            channels = []
    filenames = [f.name for f in uploaded_files] if uploaded_files else []
    return channels, filenames


# ------------------------- Core batch analysis (clean output) -------------------------
def analyze_fcs_batch(uploaded_files, dna_channel,
                      standard_filename,
                      standard_tolerance_percent=20.0,
                      min_channel=0.0, max_peaks=3,
                      n_restarts=50, max_plausible_cv=20.0):
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
                        numeric_data[dna_channel].values, min_channel=min_channel,
                        max_peaks=max_peaks, n_restarts=n_restarts, max_plausible_cv=max_plausible_cv
                    )
            except Exception:
                pass

        # Get standard reference peak from the standard file
        standard_reference_mean = None
        if standard_filename and standard_filename in fit_cache:
            std_peaks = fit_cache[standard_filename].get("peaks", [])
            if std_peaks:
                standard_reference_mean = std_peaks[0]["mu"]

        # Second pass: build summary rows
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                if filename not in parsed_cache:
                    raise ValueError("File failed to parse.")
                meta, data = parsed_cache[filename]
                numeric_data = data.select_dtypes(include="number")

                date_fcm = meta.get("$DATE", "") if isinstance(meta, dict) else ""

                # Initialize values
                mean_embryo = cv_embryo = ""
                mean_endosperm = cv_endosperm = ""
                mean_standard = cv_standard = ""
                embryo_standard = endosperm_standard = endosperm_embryo = ""

                fit = fit_cache.get(filename)
                if fit and fit["success"]:
                    peaks = fit["peaks"]  # sorted by mu

                    # Identify Standard peak based on reference mean
                    std_peak = None
                    if standard_reference_mean is not None:
                        tol = standard_tolerance_percent / 100.0
                        best_diff = np.inf
                        for p in peaks:
                            diff = abs(p["mu"] - standard_reference_mean) / standard_reference_mean
                            if diff < best_diff:
                                best_diff = diff
                                std_peak = p
                        if best_diff > tol:
                            std_peak = None

                    # Separate non-standard peaks
                    non_std_peaks = [p for p in peaks if p is not std_peak]
                    non_std_peaks.sort(key=lambda p: p["mu"])

                    embryo_peak = non_std_peaks[0] if len(non_std_peaks) >= 1 else None
                    endosperm_peak = non_std_peaks[1] if len(non_std_peaks) >= 2 else None

                    if embryo_peak:
                        mean_embryo = round(float(embryo_peak["mu"]), 3)
                        cv_embryo = round(embryo_peak["cv_percent"], 3) if embryo_peak["cv_percent"] is not None else ""
                    if endosperm_peak:
                        mean_endosperm = round(float(endosperm_peak["mu"]), 3)
                        cv_endosperm = round(endosperm_peak["cv_percent"], 3) if endosperm_peak["cv_percent"] is not None else ""
                    if std_peak:
                        mean_standard = round(float(std_peak["mu"]), 3)
                        cv_standard = round(std_peak["cv_percent"], 3) if std_peak["cv_percent"] is not None else ""

                    # Ratios
                    if mean_embryo != "" and mean_standard != "":
                        embryo_standard = round(float(mean_embryo / mean_standard), 4)
                    if mean_endosperm != "" and mean_standard != "":
                        endosperm_standard = round(float(mean_endosperm / mean_standard), 4)
                    if mean_embryo != "" and mean_endosperm != "":
                        endosperm_embryo = round(float(mean_endosperm / mean_embryo), 4)

                # Build clean row with only required columns
                row = {
                    "File": filename,
                    "Embryo Mean": mean_embryo,
                    "Embryo CV": cv_embryo,
                    "Endosperm Mean": mean_endosperm,
                    "Endosperm CV": cv_endosperm,
                    "Standard Mean": mean_standard,
                    "Standard CV": cv_standard,
                    "Embryo/Standard": embryo_standard,
                    "Endosperm/Standard": endosperm_standard,
                    "Endosperm/Embryo": endosperm_embryo,
                    "date_FCM": date_fcm,
                }
                batch_summary_rows.append(row)

            except Exception:
                # On error, still add a row with empty values
                batch_summary_rows.append({
                    "File": filename,
                    "Embryo Mean": "",
                    "Embryo CV": "",
                    "Endosperm Mean": "",
                    "Endosperm CV": "",
                    "Standard Mean": "",
                    "Standard CV": "",
                    "Embryo/Standard": "",
                    "Endosperm/Standard": "",
                    "Endosperm/Embryo": "",
                    "date_FCM": "",
                })

        batch_summary_df = pd.DataFrame(batch_summary_rows, columns=OUTPUT_COLUMNS)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            batch_summary_df.to_excel(writer, sheet_name="Analysis", index=False)
        excel_bytes = excel_buffer.getvalue()

        summary_text = f"Batch processing complete. {len(batch_summary_df)} files processed."
        return summary_text, batch_summary_df, excel_bytes

    except Exception:
        error_trace = traceback.format_exc()
        return f"Fatal error during batch processing:\n{error_trace}", pd.DataFrame(), None


# ------------------------- Streamlit UI (minimal) -------------------------
st.set_page_config(page_title="Flow Cytometry Analysis", layout="wide")
st.title("Flow Cytometry Analysis")
st.write(
    "Upload your `.fcs` files and select the internal standard file. "
    "The output Excel contains only the peak means, CVs, and ratios – no extra notes."
)

uploaded_files = st.file_uploader(
    "Upload .fcs Files", type=["fcs"], accept_multiple_files=True,
)

dna_channel = None
standard_filename = None
standard_tolerance_percent = 20.0
min_channel = 0.0
max_peaks = 3
n_restarts = 50
max_plausible_cv = 20.0

if uploaded_files:
    channels, filenames = peek_channels_and_files(uploaded_files)

    col1, col2 = st.columns(2)
    with col1:
        if channels:
            dna_channel = st.selectbox("DNA / PI Channel", options=channels)
        else:
            st.warning("Could not read channel names from the first file.")
    with col2:
        standard_filename = st.selectbox("Internal Standard file", options=filenames)

    with st.expander("Advanced settings (optional)"):
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            standard_tolerance_percent = st.number_input("Standard tolerance (%)", min_value=5.0, max_value=100.0, value=20.0, step=5.0)
        with col4:
            min_channel = st.number_input("Debris cutoff", min_value=0.0, value=0.0, step=100.0)
        with col5:
            max_peaks = st.number_input("Max peaks", min_value=1, max_value=5, value=3, step=1)
        with col6:
            n_restarts = st.number_input("Fit restarts", min_value=10, max_value=500, value=50, step=10)

        max_plausible_cv = st.number_input("Max plausible CV%", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

    # Preview (optional)
    st.subheader("Preview fit")
    preview_file = st.selectbox("File to preview", options=filenames, key="preview")
    if st.button("Show Preview") and dna_channel:
        pfile = next(f for f in uploaded_files if f.name == preview_file)
        try:
            _, pdata = _parse_fcs(pfile)
            pdata = pdata.dropna()
            pnumeric = pdata.select_dtypes(include="number")
            if dna_channel in pnumeric.columns:
                fit = fit_ploidy_peaks(
                    pnumeric[dna_channel].values, min_channel=min_channel,
                    max_peaks=max_peaks, n_restarts=n_restarts, max_plausible_cv=max_plausible_cv
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                if fit["hist"] is not None:
                    bc, h = fit["hist"]
                    ax.bar(bc, h, width=(bc[1] - bc[0]) if len(bc) > 1 else 1, alpha=0.5)
                if fit["fit_curve"] is not None:
                    xf, yf = fit["fit_curve"]
                    ax.plot(xf, yf, color="red", linewidth=2)
                ax.set_xlabel(dna_channel)
                ax.set_ylabel("Count")
                st.pyplot(fig)
                if fit["success"]:
                    for i, p in enumerate(fit["peaks"]):
                        st.write(f"Peak {i+1}: mean={p['mu']:.1f}, CV={p['cv_percent']:.2f}%")
                else:
                    st.warning("No peaks fitted.")
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
        with st.spinner("Processing..."):
            summary, df, excel = analyze_fcs_batch(
                uploaded_files, dna_channel,
                standard_filename,
                standard_tolerance_percent,
                min_channel, max_peaks, n_restarts, max_plausible_cv
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
