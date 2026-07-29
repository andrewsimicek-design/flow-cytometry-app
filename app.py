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
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

OUTPUT_XLSX_NAME = "batch_flow_cytometry_analysis.xlsx"

TRACKING_FIELDS = [
    "2_peak_CV",
    "3_peak",
    "3_peak_CV",
    "4_peak",
    "4_peak_CV",
    "raKo_sample/standard",
    "raKo_endosperm/embryo",
    "date_FCM",
    "notes",
]


# ------------------------- Gaussian fitting core -------------------------

def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _multi_gaussian(x, *params):
    """Sum of N Gaussians. params is a flat sequence of (amp, mu, sigma) triplets."""
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp, mu, sigma = params[i:i + 3]
        y += _gaussian(x, amp, mu, sigma)
    return y


def _rough_sigma(smoothed, bin_centers, idx, left_bound_idx, right_bound_idx):
    """Half-max width around a candidate peak, clipped so it can't cross into a neighboring peak."""
    amp0 = smoothed[idx]
    half = amp0 / 2.0
    li, ri = idx, idx
    while li > left_bound_idx and smoothed[li] > half:
        li -= 1
    while ri < right_bound_idx and smoothed[ri] > half:
        ri += 1
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    return max((bin_centers[ri] - bin_centers[li]) / 2.355, bin_width)


def _fit_n_gaussians(x_fit, y_fit, components, values_min, values_max, bin_width):
    """Fit a sum of Gaussians given initial guesses `components` (list of dicts with amp/mu/sigma)."""
    p0, lower, upper = [], [], []
    sigma_floor = max(bin_width * 1.5, 1e-6)
    for c in components:
        p0.extend([c["amp"], c["mu"], max(c["sigma"], sigma_floor * 2)])
        lower.extend([0, values_min, sigma_floor])
        upper.extend([np.inf, values_max, values_max])
    popt, _ = curve_fit(_multi_gaussian, x_fit, y_fit, p0=p0, bounds=(lower, upper), maxfev=30000)
    fitted = []
    for i in range(0, len(popt), 3):
        amp, mu, sigma = popt[i:i + 3]
        cv = abs(sigma / mu) * 100 if mu != 0 else None
        fitted.append({"mu": mu, "sigma": sigma, "amp": amp, "cv_percent": cv})
    return fitted, _multi_gaussian(x_fit, *popt)


def fit_ploidy_peaks(values, min_channel=0.0, bins=300, smooth_window=5,
                      min_prominence_frac=0.03, min_distance_frac=0.015, max_peaks=3,
                      residual_prominence_frac=0.06):
    """
    Detect and Gaussian-fit up to `max_peaks` ploidy peaks in a 1D array of channel values,
    using an iterative residual (peak-stripping) approach:

      1. Exclude events below `min_channel` (debris cutoff).
      2. Fit the single tallest peak first.
      3. Subtract that fit from the histogram, leaving a residual.
      4. Search the RESIDUAL for the next-largest unexplained bump. This catches small
         or off-position peaks that are hidden in another peak's shoulder/tail and
         would never show up as their own local maximum in the raw histogram.
      5. If a significant residual bump is found and max_peaks hasn't been reached, add
         it as a new component and refit ALL components together against the ORIGINAL
         histogram (not the residual) so previously-fit peaks are refined too.
      6. Repeat until no significant residual remains or max_peaks is reached.

    CV% is computed from the FITTED sigma/mu -- the standard convention in cytometry
    analysis software, robust to overlapping/off-position peaks.

    Returns a dict:
      {
        "success": bool,
        "peaks": [ {"mu": ..., "sigma": ..., "amp": ..., "cv_percent": ...}, ... ] (ascending mu),
        "hist": (bin_centers, hist_counts),
        "fit_curve": (x_smooth, y_smooth) or None,
        "note": str
      }
    """
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

    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(hist, kernel, mode="same")
    x_fit = bin_centers
    y_fit = hist.astype(float)
    n_bins = len(smoothed)

    global_max = smoothed.max()
    if global_max <= 0:
        result["note"] = "No signal in histogram."
        return result

    # --- seed with the single tallest peak ---
    seed_prom = max(global_max * min_prominence_frac, 1e-9)
    seed_dist = max(int(bins * min_distance_frac), 1)
    peak_idx, _ = find_peaks(smoothed, prominence=seed_prom, distance=seed_dist)
    if len(peak_idx) == 0:
        result["note"] = "No peaks detected."
        return result

    tallest = peak_idx[np.argmax(smoothed[peak_idx])]
    components = [{
        "amp": smoothed[tallest], "mu": bin_centers[tallest],
        "sigma": _rough_sigma(smoothed, bin_centers, tallest, 0, n_bins - 1),
    }]

    try:
        fitted, curve = _fit_n_gaussians(x_fit, y_fit, components, values.min(), values.max(), bin_width)
    except Exception as fit_err:
        result["note"] = f"Initial peak fit failed: {fit_err}"
        return result

    min_sep_bins = max(int(bins * 0.03), 3)

    # --- iteratively search the residual for additional (possibly hidden/off-position) peaks ---
    while len(fitted) < max_peaks:
        residual = y_fit - curve
        residual_smoothed = np.convolve(residual, kernel, mode="same")

        res_prom = max(global_max * residual_prominence_frac, 1e-9)
        res_idx, _ = find_peaks(residual_smoothed, prominence=res_prom, distance=seed_dist)

        # ignore residual candidates too close to an already-fitted peak
        existing_bin_positions = [np.argmin(np.abs(bin_centers - c["mu"])) for c in fitted]
        res_idx = [ri for ri in res_idx if all(abs(ri - e) >= min_sep_bins for e in existing_bin_positions)]

        if not res_idx:
            break

        best_ri = max(res_idx, key=lambda ri: residual_smoothed[ri])
        new_component = {
            "amp": max(residual_smoothed[best_ri], bin_width),
            "mu": bin_centers[best_ri],
            "sigma": _rough_sigma(residual_smoothed, bin_centers, best_ri, 0, n_bins - 1),
        }

        trial_components = [{"amp": c["amp"], "mu": c["mu"], "sigma": c["sigma"]} for c in fitted] + [new_component]
        try:
            trial_fitted, trial_curve = _fit_n_gaussians(
                x_fit, y_fit, trial_components, values.min(), values.max(), bin_width
            )
        except Exception:
            break  # adding this peak didn't help -- stop here with what we have

        # accept the extra peak only if it isn't a negligible/degenerate artifact
        max_amp = max(p["amp"] for p in trial_fitted)
        if new := next((p for p in trial_fitted if abs(p["mu"] - new_component["mu"]) < bin_width * 5), None):
            if new["amp"] < 0.04 * max_amp:
                break

        fitted, curve = trial_fitted, trial_curve

    fitted.sort(key=lambda p: p["mu"])
    result["peaks"] = fitted
    result["fit_curve"] = (x_fit, curve)
    result["success"] = True
    if len(fitted) > 1:
        result["note"] = f"Found {len(fitted)} peak(s) via iterative residual search."
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


# ------------------------- Core batch analysis -------------------------

def analyze_fcs_batch(uploaded_files, dna_channel, standard_filename, min_channel, max_peaks=3):
    try:
        if not uploaded_files:
            return "No files were uploaded. Please upload one or more .fcs files.", pd.DataFrame(), None

        batch_summary_rows = []
        combined_stats_rows = []
        combined_preview_frames = []

        success_count = 0
        failure_count = 0
        error_messages = []

        parsed_cache = {}
        fit_cache = {}

        # first pass: parse + fit every file so we know the standard's 2C peak
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                meta, data = _parse_fcs(uploaded_file)
                data = data.dropna()
                if data.empty:
                    raise ValueError("No usable event data remained after cleaning (dropna).")
                parsed_cache[filename] = (meta, data)

                numeric_data = data.select_dtypes(include="number")
                if dna_channel in numeric_data.columns:
                    fit_cache[filename] = fit_ploidy_peaks(
                        numeric_data[dna_channel].values, min_channel=min_channel, max_peaks=max_peaks
                    )
            except Exception:
                pass

        standard_2c_mean = None
        if standard_filename and standard_filename in fit_cache:
            peaks = fit_cache[standard_filename]["peaks"]
            if peaks:
                standard_2c_mean = peaks[0]["mu"]

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                if filename not in parsed_cache:
                    raise ValueError("File failed to parse.")
                meta, data = parsed_cache[filename]
                numeric_data = data.select_dtypes(include="number")

                # --- PCA ---
                pca_note = "PCA not run (insufficient channels/events)."
                if numeric_data.shape[1] >= 2 and numeric_data.shape[0] >= 2:
                    n_components = min(2, numeric_data.shape[1])
                    pca = PCA(n_components=n_components)
                    pca.fit_transform(numeric_data)
                    explained = pca.explained_variance_ratio_
                    pca_note = "; ".join(
                        f"PC{i + 1} explained variance = {ev:.3f}" for i, ev in enumerate(explained)
                    )
                    for i, ev in enumerate(explained):
                        combined_stats_rows.append({
                            "File": filename, "Channel": f"PCA_PC{i + 1}",
                            "Count": numeric_data.shape[0], "Mean": None, "Median": None,
                            "Std Dev": None, "Explained Variance Ratio": round(float(ev), 6),
                        })

                # --- per-channel stats ---
                for channel in numeric_data.columns:
                    series = numeric_data[channel]
                    combined_stats_rows.append({
                        "File": filename, "Channel": channel,
                        "Count": int(series.count()), "Mean": float(series.mean()),
                        "Median": float(series.median()), "Std Dev": float(series.std()),
                        "Explained Variance Ratio": None,
                    })

                # --- Gaussian-fitted ploidy peaks (up to 3: 2C, 3C, 4C) ---
                two_peak_cv = ""
                three_peak = ""
                three_peak_cv = ""
                four_peak = ""
                four_peak_cv = ""
                rako_endo_embryo = ""
                rako_sample_std = ""
                date_fcm = meta.get("$DATE", "") if isinstance(meta, dict) else ""

                fit = fit_cache.get(filename)
                if fit and fit["success"]:
                    peaks = fit["peaks"]
                    if len(peaks) >= 1 and peaks[0]["cv_percent"] is not None:
                        two_peak_cv = round(peaks[0]["cv_percent"], 3)
                    if len(peaks) >= 2:
                        three_peak = round(float(peaks[1]["mu"]), 3)
                        if peaks[1]["cv_percent"] is not None:
                            three_peak_cv = round(peaks[1]["cv_percent"], 3)
                        if peaks[0]["mu"] != 0:
                            rako_endo_embryo = round(float(peaks[1]["mu"] / peaks[0]["mu"]), 4)
                    if len(peaks) >= 3:
                        four_peak = round(float(peaks[2]["mu"]), 3)
                        if peaks[2]["cv_percent"] is not None:
                            four_peak_cv = round(peaks[2]["cv_percent"], 3)

                    if standard_2c_mean:
                        if filename == standard_filename:
                            rako_sample_std = 1.0
                        elif peaks:
                            my_2c = peaks[0]["mu"]
                            if standard_2c_mean != 0:
                                rako_sample_std = round(float(my_2c / standard_2c_mean), 4)

                summary_row = {
                    "File": filename,
                    "Total Events (post-cleaning)": int(numeric_data.shape[0]),
                    "Channels": ", ".join(numeric_data.columns.astype(str)),
                    "PCA Note": pca_note,
                    "2_peak_CV": two_peak_cv,
                    "3_peak": three_peak,
                    "3_peak_CV": three_peak_cv,
                    "4_peak": four_peak,
                    "4_peak_CV": four_peak_cv,
                    "raKo_sample/standard": rako_sample_std,
                    "raKo_endosperm/embryo": rako_endo_embryo,
                    "date_FCM": date_fcm,
                    "notes": "" if (fit and fit["success"]) else (fit["note"] if fit else "DNA channel not found"),
                }
                batch_summary_rows.append(summary_row)

                preview_chunk = numeric_data.head(10).copy()
                preview_chunk.insert(0, "File", filename)
                combined_preview_frames.append(preview_chunk)

                success_count += 1

            except Exception as file_err:
                failure_count += 1
                error_messages.append(f"{filename}: {file_err}")
                batch_summary_rows.append({
                    "File": filename, "Total Events (post-cleaning)": "ERROR",
                    "Channels": "", "PCA Note": f"Failed to process: {file_err}",
                    **{field: "" for field in TRACKING_FIELDS},
                })

        batch_summary_df = pd.DataFrame(batch_summary_rows)
        combined_stats_df = pd.DataFrame(combined_stats_rows)
        combined_preview_df = (
            pd.concat(combined_preview_frames, ignore_index=True) if combined_preview_frames else pd.DataFrame()
        )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            batch_summary_df.to_excel(writer, sheet_name="Batch Summary & Peaks", index=False)
            combined_stats_df.to_excel(writer, sheet_name="Combined Statistics", index=False)
            combined_preview_df.to_excel(writer, sheet_name="Processed Data Preview", index=False)
        excel_bytes = excel_buffer.getvalue()

        summary_lines = [
            f"Batch processing complete: {success_count} file(s) succeeded, {failure_count} file(s) failed "
            f"out of {len(uploaded_files)} total.",
        ]
        if error_messages:
            summary_lines.append("Errors:")
            summary_lines.extend(f"  - {msg}" for msg in error_messages)
        summary_lines.append(f"Master report ready for download: {OUTPUT_XLSX_NAME}")
        summary_text = "\n".join(summary_lines)

        return summary_text, batch_summary_df, excel_bytes

    except Exception:
        error_trace = traceback.format_exc()
        return f"A fatal error occurred during batch processing:\n{error_trace}", pd.DataFrame(), None


# ------------------------- Streamlit UI -------------------------

st.set_page_config(page_title="Flow Cytometry Batch Analysis", layout="wide")
st.title("Flow Cytometry Batch Analysis")
st.write(
    "Upload one or more .fcs files. The app fits Gaussian curve(s) to the ploidy peak(s) on your "
    "chosen DNA-fluorescence channel -- the same approach used by standard flow cytometry analysis "
    "software -- and reports fitted peak position, CV%, endosperm/embryo ratio, and (if you designate "
    "an internal standard) the sample/standard ratio."
)

uploaded_files = st.file_uploader(
    "Upload .fcs Flow Cytometry Files (batch)", type=["fcs"], accept_multiple_files=True,
)

dna_channel = None
standard_filename = "None"
min_channel = 0.0
max_peaks = 3

if uploaded_files:
    channels, filenames = peek_channels_and_files(uploaded_files)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if channels:
            dna_channel = st.selectbox(
                "DNA / PI Fluorescence Channel",
                options=channels,
                help="Channel representing DNA content / PI fluorescence.",
            )
        else:
            st.warning("Could not read channel names from the first file.")
    with col2:
        standard_filename = st.selectbox(
            "Internal Standard file (optional)", options=["None"] + filenames,
        )
    with col3:
        min_channel = st.number_input(
            "Debris cutoff (exclude events below this channel value)",
            min_value=0.0, value=0.0, step=100.0,
            help="Set this above any debris peak near the origin, based on the preview plot below.",
        )
    with col4:
        max_peaks = st.number_input(
            "Max peaks to detect",
            min_value=1, max_value=5, value=3, step=1,
            help="Detects up to this many peaks (e.g. 2C, 3C, 4C), even if some are small or close together.",
        )

    st.subheader("Preview: check the fit before running the full batch")
    preview_file_name = st.selectbox("File to preview", options=filenames, key="preview_select")
    if st.button("Generate Preview") and dna_channel:
        preview_file = next(f for f in uploaded_files if f.name == preview_file_name)
        try:
            _, pdata = _parse_fcs(preview_file)
            pdata = pdata.dropna()
            pnumeric = pdata.select_dtypes(include="number")
            if dna_channel not in pnumeric.columns:
                st.error(f"Channel '{dna_channel}' not found in this file.")
            else:
                fit = fit_ploidy_peaks(pnumeric[dna_channel].values, min_channel=min_channel, max_peaks=max_peaks)
                fig, ax = plt.subplots(figsize=(8, 4))
                if fit["hist"] is not None:
                    bc, h = fit["hist"]
                    ax.bar(bc, h, width=(bc[1] - bc[0]) if len(bc) > 1 else 1, alpha=0.5, label="Histogram")
                if fit["fit_curve"] is not None:
                    xf, yf = fit["fit_curve"]
                    ax.plot(xf, yf, color="red", linewidth=2, label="Gaussian fit")
                ax.set_xlabel(dna_channel)
                ax.set_ylabel("Event count")
                ax.legend()
                st.pyplot(fig)

                if fit["success"]:
                    for i, p in enumerate(fit["peaks"]):
                        cv_txt = f"{p['cv_percent']:.2f}%" if p["cv_percent"] is not None else "N/A"
                        st.write(f"Peak {i + 1}: mu={p['mu']:.1f}, CV={cv_txt}")
                else:
                    st.warning(fit["note"])
        except Exception as preview_err:
            st.error(f"Preview failed: {preview_err}")

if st.button("Run Batch Analysis", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one .fcs file first.")
    elif not dna_channel:
        st.warning("Please select a DNA/PI fluorescence channel first.")
    else:
        std_name = None if standard_filename == "None" else standard_filename
        with st.spinner("Processing files..."):
            summary_text, preview_df, excel_bytes = analyze_fcs_batch(
                uploaded_files, dna_channel, std_name, min_channel, max_peaks
            )

        st.subheader("Processing Summary")
        st.text(summary_text)

        st.subheader("Batch Summary Preview")
        st.dataframe(preview_df, use_container_width=True)

        if excel_bytes is not None:
            st.subheader("Download Master Excel Report")
            st.download_button(
                label=f"Download {OUTPUT_XLSX_NAME}",
                data=excel_bytes,
                file_name=OUTPUT_XLSX_NAME,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
