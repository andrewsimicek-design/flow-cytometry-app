import os
import io
import tempfile
import traceback

import numpy as np
import streamlit as st
import pandas as pd
import fcsparser
from sklearn.decomposition import PCA
from scipy.signal import find_peaks

OUTPUT_XLSX_NAME = "batch_flow_cytometry_analysis.xlsx"

TRACKING_FIELDS = [
    "2_peak_CV",
    "3_peak",
    "3_peak_CV",
    "raKo_sample/standard",
    "raKo_endosperm/embryo",
    "date_FCM",
    "notes",
]


# ------------------------- Helpers -------------------------

def _parse_fcs(uploaded_file):
    """Write an in-memory uploaded file to disk temporarily and parse it with fcsparser."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fcs") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        meta, data = fcsparser.parse(tmp_path, reformat_meta=True)
    finally:
        os.remove(tmp_path)
    return meta, data


def _detect_peaks(values, bins=200, smooth_window=5, min_prominence_frac=0.05, min_distance_frac=0.03):
    """
    Detect up to two dominant peaks (ploidy peaks) in a 1D array of channel values.
    Returns a list of (peak_center_value, group_values) for each detected peak,
    sorted by ascending channel position (lower = 2C/embryo, higher = 3C/endosperm).
    """
    values = np.asarray(values, dtype=float)
    if values.size < 10:
        return []

    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # simple moving-average smoothing
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(hist, kernel, mode="same")
    else:
        smoothed = hist.astype(float)

    prominence = max(smoothed.max() * min_prominence_frac, 1e-9)
    distance = max(int(bins * min_distance_frac), 1)

    peak_idx, props = find_peaks(smoothed, prominence=prominence, distance=distance)
    if len(peak_idx) == 0:
        return []

    # Keep the two tallest peaks, then order them by position (ascending)
    heights = smoothed[peak_idx]
    top_two_idx = peak_idx[np.argsort(heights)[-2:]]
    top_two_idx = np.sort(top_two_idx)

    if len(top_two_idx) == 1:
        center = bin_centers[top_two_idx[0]]
        return [(center, values)]

    # two peaks -> split at the valley (minimum) between them
    i1, i2 = top_two_idx
    valley_local_idx = i1 + np.argmin(smoothed[i1:i2 + 1])
    valley_value = bin_centers[valley_local_idx]

    group1 = values[values <= valley_value]
    group2 = values[values > valley_value]

    if group1.size == 0 or group2.size == 0:
        # fallback: treat as a single dominant peak
        center = bin_centers[top_two_idx[np.argmax(heights)]]
        return [(center, values)]

    center1 = bin_centers[i1]
    center2 = bin_centers[i2]
    return [(center1, group1), (center2, group2)]


def _cv_percent(group_values):
    mean = float(np.mean(group_values))
    std = float(np.std(group_values))
    if mean == 0:
        return None
    return (std / mean) * 100.0


def peek_channels_and_files(uploaded_files):
    """Parse just the first file (lightweight) to list available numeric channels,
    and return the full filename list for standard-selection."""
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


# ------------------------- Core analysis -------------------------

def analyze_fcs_batch(uploaded_files, dna_channel, standard_filename):
    try:
        if not uploaded_files:
            return "No files were uploaded. Please upload one or more .fcs files.", pd.DataFrame(), None

        batch_summary_rows = []
        combined_stats_rows = []
        combined_preview_frames = []

        # first pass: compute 2C-peak mean for every file (needed for sample/standard ratio)
        peak2c_means = {}

        success_count = 0
        failure_count = 0
        error_messages = []

        parsed_cache = {}

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
                    peaks = _detect_peaks(numeric_data[dna_channel].values)
                    if peaks:
                        peak2c_means[filename] = peaks[0][0]
            except Exception:
                pass  # handled fully in the main loop below

        standard_2c_mean = peak2c_means.get(standard_filename) if standard_filename else None

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

                # --- ploidy peak detection on the chosen DNA channel ---
                two_peak_cv = ""
                three_peak = ""
                three_peak_cv = ""
                rako_endo_embryo = ""
                rako_sample_std = ""
                date_fcm = meta.get("$DATE", "") if isinstance(meta, dict) else ""

                if dna_channel in numeric_data.columns:
                    peaks = _detect_peaks(numeric_data[dna_channel].values)
                    if len(peaks) >= 1:
                        c1, g1 = peaks[0]
                        cv1 = _cv_percent(g1)
                        two_peak_cv = round(cv1, 3) if cv1 is not None else ""
                    if len(peaks) == 2:
                        c2, g2 = peaks[1]
                        cv2 = _cv_percent(g2)
                        three_peak = round(float(np.mean(g2)), 3)
                        three_peak_cv = round(cv2, 3) if cv2 is not None else ""
                        if np.mean(g1) != 0:
                            rako_endo_embryo = round(float(np.mean(g2) / np.mean(g1)), 4)

                    if standard_2c_mean and filename != standard_filename:
                        my_2c = peak2c_means.get(filename)
                        if my_2c and standard_2c_mean != 0:
                            rako_sample_std = round(float(my_2c / standard_2c_mean), 4)
                    elif filename == standard_filename:
                        rako_sample_std = 1.0

                summary_row = {
                    "File": filename,
                    "Total Events (post-cleaning)": int(numeric_data.shape[0]),
                    "Channels": ", ".join(numeric_data.columns.astype(str)),
                    "PCA Note": pca_note,
                    "2_peak_CV": two_peak_cv,
                    "3_peak": three_peak,
                    "3_peak_CV": three_peak_cv,
                    "raKo_sample/standard": rako_sample_std,
                    "raKo_endosperm/embryo": rako_endo_embryo,
                    "date_FCM": date_fcm,
                    "notes": "",
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
    "Upload one or more .fcs files. The app auto-detects the two ploidy peaks (2C/embryo and "
    "3C/endosperm) on your chosen DNA-fluorescence channel, computes their CV%, the "
    "endosperm/embryo ratio, and (if you designate an internal standard) the sample/standard ratio."
)

uploaded_files = st.file_uploader(
    "Upload .fcs Flow Cytometry Files (batch)", type=["fcs"], accept_multiple_files=True,
)

dna_channel = None
standard_filename = "None"

if uploaded_files:
    channels, filenames = peek_channels_and_files(uploaded_files)

    col1, col2 = st.columns(2)
    with col1:
        if channels:
            dna_channel = st.selectbox(
                "DNA / PI Fluorescence Channel (used for peak detection)",
                options=channels,
                help="Pick the channel that represents DNA content / propidium iodide fluorescence, e.g. FL2-A, FL3-A, PE-A.",
            )
        else:
            st.warning("Could not read channel names from the first file. Check that it's a valid .fcs file.")
    with col2:
        standard_filename = st.selectbox(
            "Internal Standard file (optional, for sample/standard ratio)",
            options=["None"] + filenames,
        )

if st.button("Run Batch Analysis", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one .fcs file first.")
    elif not dna_channel:
        st.warning("Please select a DNA/PI fluorescence channel first.")
    else:
        std_name = None if standard_filename == "None" else standard_filename
        with st.spinner("Processing files..."):
            summary_text, preview_df, excel_bytes = analyze_fcs_batch(uploaded_files, dna_channel, std_name)

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
