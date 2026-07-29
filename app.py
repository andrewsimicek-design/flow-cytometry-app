import os
import io
import tempfile
import traceback

import streamlit as st
import pandas as pd
import fcsparser
from sklearn.decomposition import PCA

OUTPUT_XLSX_NAME = "batch_flow_cytometry_analysis.xlsx"

# Manual/experimental tracking fields required on the "Batch Summary & Peaks" tab.
# Left blank for the user to fill in after visually inspecting each sample's
# histogram/peaks, since true peak-calling requires manual gating review.
TRACKING_FIELDS = [
    "2_peak_CV",
    "3_peak",
    "3_peak_CV",
    "raKo_sample/standard",
    "raKo_endosperm/embryo",
    "date_FCM",
    "notes",
]


def analyze_fcs_batch(uploaded_files):
    """
    Batch-process a list of uploaded .fcs files:
      - parse each file with fcsparser
      - clean event data (dropna)
      - run PCA on numeric event data when applicable
      - compute per-channel summary statistics
      - build a single master Excel workbook (in-memory) with 3 tabs

    Returns:
        summary_text (str), batch_summary_df (pd.DataFrame), excel_bytes (bytes)
    """
    try:
        if not uploaded_files:
            return "No files were uploaded. Please upload one or more .fcs files.", pd.DataFrame(), None

        batch_summary_rows = []       # Tab 1: Batch Summary & Peaks
        combined_stats_rows = []      # Tab 2: Combined Statistics
        combined_preview_frames = []  # Tab 3: Processed Data Preview

        success_count = 0
        failure_count = 0
        error_messages = []

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name

            try:
                # Streamlit gives us an in-memory UploadedFile, but fcsparser
                # needs a real filesystem path -> write to a temp file first.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fcs") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    # --- Parse the .fcs file ---
                    meta, data = fcsparser.parse(tmp_path, reformat_meta=True)
                finally:
                    os.remove(tmp_path)

                # --- Clean the event data ---
                data = data.dropna()

                if data.empty:
                    raise ValueError("No usable event data remained after cleaning (dropna).")

                numeric_data = data.select_dtypes(include="number")

                # --- Run PCA on the events, if applicable ---
                pca_note = "PCA not run (insufficient channels/events)."
                if numeric_data.shape[1] >= 2 and numeric_data.shape[0] >= 2:
                    n_components = min(2, numeric_data.shape[1])
                    pca = PCA(n_components=n_components)
                    pca.fit_transform(numeric_data)
                    explained = pca.explained_variance_ratio_
                    pca_note = "; ".join(
                        f"PC{i + 1} explained variance = {ev:.3f}"
                        for i, ev in enumerate(explained)
                    )
                    for i, ev in enumerate(explained):
                        combined_stats_rows.append({
                            "File": filename,
                            "Channel": f"PCA_PC{i + 1}",
                            "Count": numeric_data.shape[0],
                            "Mean": None,
                            "Median": None,
                            "Std Dev": None,
                            "Explained Variance Ratio": round(float(ev), 6),
                        })

                # --- Per-channel summary statistics (Count, Mean, Median, Std Dev) ---
                for channel in numeric_data.columns:
                    series = numeric_data[channel]
                    combined_stats_rows.append({
                        "File": filename,
                        "Channel": channel,
                        "Count": int(series.count()),
                        "Mean": float(series.mean()),
                        "Median": float(series.median()),
                        "Std Dev": float(series.std()),
                        "Explained Variance Ratio": None,
                    })

                # --- Batch Summary & Peaks row (manual tracking fields left blank) ---
                summary_row = {
                    "File": filename,
                    "Total Events (post-cleaning)": int(numeric_data.shape[0]),
                    "Channels": ", ".join(numeric_data.columns.astype(str)),
                    "PCA Note": pca_note,
                }
                for field in TRACKING_FIELDS:
                    summary_row[field] = ""  # left blank for manual entry
                batch_summary_rows.append(summary_row)

                # --- Processed data preview (first few rows, tagged with filename) ---
                preview_chunk = numeric_data.head(10).copy()
                preview_chunk.insert(0, "File", filename)
                combined_preview_frames.append(preview_chunk)

                success_count += 1

            except Exception as file_err:
                failure_count += 1
                error_messages.append(f"{filename}: {file_err}")
                batch_summary_rows.append({
                    "File": filename,
                    "Total Events (post-cleaning)": "ERROR",
                    "Channels": "",
                    "PCA Note": f"Failed to process: {file_err}",
                    **{field: "" for field in TRACKING_FIELDS},
                })

        # --- Assemble DataFrames for each tab ---
        batch_summary_df = pd.DataFrame(batch_summary_rows)
        combined_stats_df = pd.DataFrame(combined_stats_rows)
        combined_preview_df = (
            pd.concat(combined_preview_frames, ignore_index=True)
            if combined_preview_frames else pd.DataFrame()
        )

        # --- Write the master Excel report to an in-memory buffer ---
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            batch_summary_df.to_excel(writer, sheet_name="Batch Summary & Peaks", index=False)
            combined_stats_df.to_excel(writer, sheet_name="Combined Statistics", index=False)
            combined_preview_df.to_excel(writer, sheet_name="Processed Data Preview", index=False)
        excel_bytes = excel_buffer.getvalue()

        # --- Build summary text ---
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
    "Upload one or more .fcs files. Each file is parsed, cleaned, and analyzed "
    "(including PCA on event channels). A single master Excel workbook is generated "
    "with three tabs: Batch Summary & Peaks, Combined Statistics, and Processed Data Preview."
)

uploaded_files = st.file_uploader(
    "Upload .fcs Flow Cytometry Files (batch)",
    type=["fcs"],
    accept_multiple_files=True,
)

if st.button("Run Batch Analysis", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one .fcs file first.")
    else:
        with st.spinner("Processing files..."):
            summary_text, preview_df, excel_bytes = analyze_fcs_batch(uploaded_files)

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
