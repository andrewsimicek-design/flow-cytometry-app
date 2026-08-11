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


def _make_model_with_background(x_min):
    """
    Build a model function: exponential-decay BACKGROUND + sum of N Gaussian PEAKS.

    The background term absorbs broad, slowly-decaying debris/noise baseline (very
    common in flow cytometry histograms: lots of small events at low channels, tailing
    off toward higher channels). Without it, that baseline "hump" competes with genuine
    small peaks for a limited peak budget and can get modeled as a fake broad peak
    instead -- which was hiding real, smaller peaks (e.g. a real 3rd peak) from being found.
    """
    def _model(x, bg_amp, bg_k, *gauss_params):
        bg = bg_amp * np.exp(-bg_k * (x - x_min))
        return bg + _multi_gaussian(x, *gauss_params)
    return _model


def _fit_fixed_n(x_fit, y_fit, n_peaks, values_min, values_max, bin_width, max_hist, n_restarts, rng):
    """
    Try to fit exactly `n_peaks` Gaussian PEAKS plus one background term to the histogram,
    using `n_restarts` random initial guesses (random peak positions/widths/heights/background
    shape each time), keeping whichever restart converges to the lowest sum-of-squared-error
    fit. This is what lets small or off-position peaks get found even when a single "smart
    guess" attempt would miss them -- across many random starting points, at least one is
    likely to land near each true peak, and the background term keeps debris/baseline noise
    from stealing a peak slot.
    """
    span = max(values_max - values_min, 1e-6)
    sigma_floor = max(bin_width * 1.2, 1e-6)
    model_func = _make_model_with_background(values_min)
    best = None  # (sse, peaks, curve, bg_amp, bg_k)

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
    """
    Merge fitted peaks that are really the same underlying population split into two
    near-duplicate Gaussians (can happen when the optimizer finds a slightly-lower-SSE
    solution by "cheating" with two overlapping components instead of one honest peak).
    Two peaks are merged if their centers are closer than `min_separation_sigma` times
    their average sigma. Merging uses proper Gaussian-mixture moment matching so the
    combined peak's mu/sigma/amp represent the pooled population, not just one of the two.
    """
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
    """
    Detect and Gaussian-fit up to `max_peaks` ploidy peaks in a 1D array of channel values
    using a multi-start global search, with a separate background term for debris/baseline:

      1. Exclude events below `min_channel` (debris cutoff).
      2. The fit model is: exponential-decay BACKGROUND + N Gaussian PEAKS. The background
         absorbs broad, low, slowly-decaying debris noise so it can't masquerade as (or hide)
         a real peak.
      3. For EVERY candidate peak count n = 1, 2, ..., max_peaks: try `n_restarts` random
         initial guesses (random positions/widths/heights/background shape) and keep the
         best-converging fit for that n. This brute-force restart strategy is what catches
         small or off-position peaks a single "smart initial guess" attempt would miss.
      4. Merge near-duplicate peaks within each candidate (same population split in two).
      5. QUALITY CONTROL: reject any candidate peak-count whose result contains a peak with
         CV% above `max_plausible_cv` -- this is the telltale sign of debris/noise being
         misfit as a fake peak (happens when max_peaks is set higher than the number of real
         populations actually present, e.g. a single-population sample with max_peaks=3).
         Only candidates that pass this check are considered further.
      6. Compare the surviving candidates using BIC (Bayesian Information Criterion), which
         rewards a lower fitting error but penalizes extra peaks, and report the winner.
         If NO candidate passes the quality check, falls back to the best unfiltered result
         and flags this clearly in the returned note.

    CV% is computed from the FITTED sigma/mu -- the standard convention in cytometry
    analysis software, robust to overlapping/off-position peaks.

    A fixed random seed is used by default so repeated runs on the same data give the
    same result (important for reproducible scientific reporting).

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

    x_fit = bin_centers
    y_fit = hist.astype(float)
    max_hist = float(hist.max())
    n_data = len(y_fit)

    if max_hist <= 0:
        result["note"] = "No signal in histogram."
        return result

    rng = np.random.default_rng(seed)

    # Fit every candidate peak count, merging near-duplicates within each one immediately.
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
        k = 3 * n_peaks + 2  # 3 params/peak + 2 background params, present in every model
        return n_data * np.log(sse / n_data) + k * np.log(n_data)

    plausible_ns = [n for n, c in candidates.items() if c["plausible"]]
    pool = plausible_ns if plausible_ns else list(candidates.keys())

    best_n = min(pool, key=lambda n: bic(candidates[n]["sse"], n))
    chosen = candidates[best_n]
    sse, peaks, curve, bg_amp, bg_k = chosen["sse"], chosen["peaks"], chosen["curve"], chosen["bg_amp"], chosen["bg_k"]

    result["peaks"] = peaks
    result["fit_curve"] = (x_fit, curve)
    result["success"] = True
    merge_note = " (2 near-duplicate peaks merged into 1)" if len(peaks) < best_n else ""
    qc_note = ""
    if not plausible_ns:
        qc_note = " QUALITY WARNING: no peak-count option passed the plausibility check (CV too high) -- showing best available anyway."
    elif len(plausible_ns) < len(candidates):
        rejected = sorted(set(candidates.keys()) - set(plausible_ns))
        qc_note = f" (rejected implausible fit(s) at peak-count {rejected} -- CV% too high, likely debris misfit as a peak)"
    result["note"] = (
        f"Selected {best_n}-peak fit, {len(peaks)} peak(s) after cleanup{merge_note}{qc_note} "
        f"(multi-start search, {n_restarts} restarts per peak count, "
        f"best of {list(candidates.keys())} peak-count options via BIC; background amp={bg_amp:.1f}, "
        f"decay k={bg_k:.5f})."
        if len(candidates) > 1 or best_n > 1 or qc_note else ""
    )
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

def analyze_fcs_batch(uploaded_files, dna_channel, standard_filename, min_channel, max_peaks=3,
                       n_restarts=50, max_plausible_cv=20.0):
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
                        numeric_data[dna_channel].values, min_channel=min_channel,
                        max_peaks=max_peaks, n_restarts=n_restarts, max_plausible_cv=max_plausible_cv
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
n_restarts = 50
max_plausible_cv = 20.0

if uploaded_files:
    channels, filenames = peek_channels_and_files(uploaded_files)

    col1, col2, col3, col4, col5 = st.columns(5)
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
    with col5:
        n_restarts = st.number_input(
            "Fit thoroughness (restarts)",
            min_value=10, max_value=500, value=50, step=10,
            help=(
                "Number of random re-attempts per peak count. Higher = more likely to find "
                "small/off-position peaks, but slower (especially with many files in a batch)."
            ),
        )

    with st.expander("Advanced: peak quality control"):
        max_plausible_cv = st.number_input(
            "Max plausible peak CV% (reject fits above this)",
            min_value=1.0, max_value=100.0, value=20.0, step=1.0,
            help=(
                "If a peak's fitted CV% exceeds this, that peak-count option is automatically "
                "rejected and the app falls back to fewer peaks -- this is what prevents debris/noise "
                "from being misfit as a fake extra peak on single-population samples "
                "(e.g. 'only_endosperm' or 'only_embryo' files). Lower this if you want stricter "
                "quality control; raise it if genuinely noisy samples are being rejected too "
                "aggressively."
            ),
        )

    with st.expander("Advanced: peak quality control"):
        max_plausible_cv = st.number_input(
            "Max plausible peak CV% (reject fits above this)",
            min_value=1.0, max_value=100.0, value=20.0, step=1.0,
            help=(
                "Real ploidy peaks are almost always well under this. Any candidate peak-count "
                "whose fit includes a peak above this CV% is rejected as likely debris/noise "
                "misfit as a fake peak, and the app automatically falls back to fewer peaks. "
                "If a file genuinely should have a noisier real peak, raise this value for that batch."
            ),
        )
        st.caption(
            "If you set 'Max peaks to detect' higher than the number of real populations a sample "
            "actually has (e.g. a single-population 'only endosperm' or 'only embryo' file), the "
            "leftover peak slot can otherwise get spent fitting debris/noise instead of being left "
            "unused. This filter catches that automatically -- check the batch summary's PCA Note / "
            "fit note column if a file's peak count looks lower than expected, it'll explain why."
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
                fit = fit_ploidy_peaks(
                    pnumeric[dna_channel].values, min_channel=min_channel,
                    max_peaks=max_peaks, n_restarts=n_restarts, max_plausible_cv=max_plausible_cv
                )
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

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if st.button("Run Batch Analysis", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one .fcs file first.")
    elif not dna_channel:
        st.warning("Please select a DNA/PI fluorescence channel first.")
    else:
        std_name = None if standard_filename == "None" else standard_filename
        with st.spinner("Processing files..."):
            summary_text, preview_df, excel_bytes = analyze_fcs_batch(
                uploaded_files, dna_channel, std_name, min_channel, max_peaks, n_restarts, max_plausible_cv
            )
        # store in session_state so results (incl. the download button) survive
        # Streamlit reruns -- e.g. the app waking up from sleep, or any other
        # widget interaction -- instead of disappearing after the button's
        # one-shot "clicked" state resets on the next rerun.
        st.session_state.batch_results = (summary_text, preview_df, excel_bytes)

if st.session_state.batch_results is not None:
    summary_text, preview_df, excel_bytes = st.session_state.batch_results

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
        if st.button("Clear results"):
            st.session_state.batch_results = None
            st.rerun()
