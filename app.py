def analyze_fcs_batch(uploaded_files, dna_channel,
                      standard_filename,
                      standard_tolerance_percent=30.0,
                      min_channel=0.0,
                      n_peaks=3, n_restarts=100, max_plausible_cv=30.0,
                      scale_to_1024=True, raw_max=32768,
                      manual_standard_mean=None,
                      force_lowest_peak_as_standard=True):
    try:
        if not uploaded_files:
            return "No files uploaded.", pd.DataFrame()

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

        # Get standard reference
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

        # Second pass: build rows
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                if filename not in parsed_cache:
                    raise ValueError("File failed to parse.")
                meta, data = parsed_cache[filename]
                numeric_data = data.select_dtypes(include="number")

                fit = fit_cache.get(filename)
                
                # Initialize with empty values
                standard_mean = standard_cv = ""
                embryo_mean = embryo_cv = ""
                endosperm_mean = endosperm_cv = ""
                embryo_standard = endosperm_standard = endosperm_embryo = ""

                if fit and fit["success"]:
                    all_peaks = fit["peaks"]  # sorted by mu

                    # ---- Identify Standard peak ----
                    std_peak = None
                    std_idx = -1

                    if force_lowest_peak_as_standard:
                        std_peak = all_peaks[0]
                        std_idx = 0
                    elif standard_reference_mean is not None:
                        tol = standard_tolerance_percent / 100.0
                        best_diff = np.inf
                        for idx, p in enumerate(all_peaks):
                            diff = abs(p["mu"] - standard_reference_mean) / standard_reference_mean
                            if diff < best_diff:
                                best_diff = diff
                                std_peak = p
                                std_idx = idx
                        if best_diff > tol:
                            std_peak = all_peaks[0]
                            std_idx = 0
                    else:
                        std_peak = all_peaks[0]
                        std_idx = 0

                    # Remove standard, label remaining by order
                    non_std = [p for i, p in enumerate(all_peaks) if i != std_idx]
                    non_std.sort(key=lambda p: p["mu"])

                    # ---- Assign values ----
                    # Standard
                    if std_peak:
                        standard_mean = round(float(std_peak["mu"]), 3)
                        standard_cv = round(std_peak["cv_percent"], 3) if std_peak["cv_percent"] is not None else ""

                    # Embryo = first non-standard
                    if len(non_std) >= 1:
                        embryo_mean = round(float(non_std[0]["mu"]), 3)
                        embryo_cv = round(non_std[0]["cv_percent"], 3) if non_std[0]["cv_percent"] is not None else ""

                    # Endosperm = second non-standard
                    if len(non_std) >= 2:
                        endosperm_mean = round(float(non_std[1]["mu"]), 3)
                        endosperm_cv = round(non_std[1]["cv_percent"], 3) if non_std[1]["cv_percent"] is not None else ""

                    # ---- Ratios ----
                    if embryo_mean != "" and standard_mean != "":
                        embryo_standard = round(float(embryo_mean / standard_mean), 4)
                    if endosperm_mean != "" and standard_mean != "":
                        endosperm_standard = round(float(endosperm_mean / standard_mean), 4)
                    if embryo_mean != "" and endosperm_mean != "":
                        endosperm_embryo = round(float(endosperm_mean / embryo_mean), 4)

                # ---- Build row in docent's format ----
                row = {
                    "Sample_ID": "",  # can be filled manually
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
                    "": "",  # empty column for spacing
                    "Date_of_analyses": today_date,
                }
                batch_summary_rows.append(row)

            except Exception:
                batch_summary_rows.append({
                    "Sample_ID": "",
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
                })

        # Create DataFrame with docent's exact column order
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
            "Date_of_analyses"
        ]

        df = pd.DataFrame(batch_summary_rows, columns=docent_columns)

        # Create Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Analysis", index=False)
        excel_bytes = excel_buffer.getvalue()

        summary_text = f"Batch complete. {len(df)} files processed."
        return summary_text, df, excel_bytes

    except Exception:
        error_trace = traceback.format_exc()
        return f"Error:\n{error_trace}", pd.DataFrame(), None
