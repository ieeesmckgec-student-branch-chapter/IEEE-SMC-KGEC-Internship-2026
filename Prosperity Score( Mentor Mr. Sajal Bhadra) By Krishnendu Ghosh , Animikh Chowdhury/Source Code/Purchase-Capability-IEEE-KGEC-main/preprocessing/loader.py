import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

COMMON_DATE_KEYS = [
    "date", "txn_date", "transaction_date", "value date", "posting date",
    "posting_date", "transaction date", "value_date", "date/time", "date time", "transaction_date_time"
]
COMMON_AMOUNT_KEYS = [
    "amount", "amt", "credit", "debit", "withdrawal", "deposit",
    "debit amount", "credit amount", "amount credited", "amount debited"
]
COMMON_DESC_KEYS = ["description", "narration", "remarks", "particulars", "details"]
COMMON_PAYEE_KEYS = ["payee", "beneficiary", "counterparty"]


def _clean_amount(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    # remove currency symbols & spaces, keep minus and dot
    s = re.sub(r"[^\d\.\-]", "", s)
    if s == "" or s == "-" or s == ".":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _is_excel_serial_like(series: pd.Series) -> float:
    """Return fraction of values that look like Excel date serials (numeric in plausible range)."""
    numeric = pd.to_numeric(series, errors="coerce")
    # plausible serials for Excel: between ~20000 and 50000 (roughly 1954..2136)
    mask = (numeric >= 20000) & (numeric <= 50000)
    if numeric.size == 0:
        return 0.0
    return float(mask.sum() / numeric.size)


def _parse_dates(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """
    Safe wrapper around pd.to_datetime that avoids passing `infer_datetime_format`
    (removed in newer pandas). Returns a datetime Series with invalid parses as NaT.
    """
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    except TypeError:
        # Defensive fallback: call without dayfirst if unexpected signature
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return pd.Series([pd.NaT] * len(series))


def _fraction_parsable_dates(series: pd.Series) -> float:
    """Attempt to parse series to datetime; return fraction successfully parsed."""
    parsed = _parse_dates(series, dayfirst=False)
    if parsed.size == 0:
        return 0.0
    return float(parsed.notna().sum() / parsed.size)


def _score_header(cols: List[str]) -> int:
    keys = " ".join([c.lower() for c in cols])
    score = 0
    if any(k in keys for k in COMMON_DATE_KEYS):
        score += 1
    if any(k in keys for k in COMMON_AMOUNT_KEYS):
        score += 1
    return score


def _find_best_header_for_file(path: str, max_header: int = 10) -> Tuple[int, Optional[pd.DataFrame]]:
    """
    Try header rows 0..max_header and pick the header that looks best by:
      - header name containing date/amount tokens
      - OR columns whose content parses as dates (high fraction)
    Returns (best_header_row, sample_df) or (0, df) if none found.
    """
    best_score = -1.0
    best_h = None
    best_sample = None
    for h in range(0, max_header + 1):
        try:
            sample = pd.read_excel(path, header=h, nrows=20, dtype=str)
        except Exception:
            continue
        cols = list(sample.columns)
        name_score = _score_header(cols)
        # content score: best fraction parsable in any column
        content_scores = []
        for c in cols:
            s = sample[c]
            frac_parsable = _fraction_parsable_dates(s)
            frac_excel = _is_excel_serial_like(s)
            content_scores.append(max(frac_parsable, frac_excel))
        content_best = max(content_scores) if content_scores else 0.0
        combined = name_score + content_best  # name_score is 0/1/2, content_best in 0..1
        if combined > best_score:
            best_score = combined
            best_h = h
            best_sample = sample
        # short circuit: perfect header found
        if best_score >= 2.0:
            break
    if best_h is None:
        # fallback
        return 0, None
    return int(best_h), best_sample


def _convert_excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    """Convert numeric Excel serials to timestamps (handles floats/ints)."""
    numeric = pd.to_numeric(series, errors="coerce")
    base = pd.Timestamp("1899-12-30")  # Excel serial base (handles Excel leap-year bug for most use cases)
    return pd.to_timedelta(numeric, unit="D") + base


def load_transaction_files(paths: List[str]) -> pd.DataFrame:
    """
    Load multiple CSV/XLSX files and normalize to columns:
    date, amount, description (opt), payee (opt), running_balance (opt)

    This function:
      - tries header rows 0..10 for Excel files and picks the most likely header
      - auto-detects the date column by name or by content (parsable dates or excel-serial-like numbers)
      - normalizes amount and running_balance to float
      - returns concatenated DataFrame with standard columns
    """
    all_dfs = []
    for p in paths:
        p = str(p)
        ext = Path(p).suffix.lower()
        if ext not in [".xlsx", ".xls", ".csv"]:
            logger.warning("Skipping unsupported file type: %s", p)
            continue

        if ext == ".csv":
            try:
                df_read = pd.read_csv(p, dtype=str)
                header_row = 0
            except Exception as e:
                logger.exception("Failed to read CSV %s: %s", p, e)
                continue
        else:
            # find best header row and sample
            best_h, _ = _find_best_header_for_file(p, max_header=10)
            try:
                df_read = pd.read_excel(p, header=best_h, dtype=str)
            except Exception as e:
                logger.exception("Failed to read Excel %s with header=%s: %s", p, best_h, e)
                # fallback to header=0
                df_read = pd.read_excel(p, header=0, dtype=str)

        # Normalize column labels (strip whitespace and non-breaking spaces)
        df_read.columns = [str(c).strip().replace("\xa0", " ") for c in df_read.columns]
        cols_lower = [c.lower() for c in df_read.columns]

        # Attempt to locate date column by name first
        date_col = None
        for i, c in enumerate(cols_lower):
            if any(k in c for k in COMMON_DATE_KEYS):
                date_col = df_read.columns[i]
                break

        # locate amount/debit/credit by name
        amount_col = None
        debit_col = None
        credit_col = None
        for i, c in enumerate(cols_lower):
            if "amount" == c or c.startswith("amount"):
                amount_col = df_read.columns[i]
            if any(k in c for k in ["debit", "withdrawal"]):
                debit_col = df_read.columns[i]
            if any(k in c for k in ["credit", "deposit"]):
                credit_col = df_read.columns[i]

        # If name-based date_col not found, use content-based detection
        if date_col is None:
            # evaluate each column's fraction parsable as date or excel serial-like
            best_frac = 0.0
            best_col = None
            for c in df_read.columns:
                frac_date = _fraction_parsable_dates(df_read[c])
                frac_serial = _is_excel_serial_like(df_read[c])
                frac = max(frac_date, frac_serial)
                if frac > best_frac:
                    best_frac = frac
                    best_col = c
            # require a reasonable fraction (>= 0.45) before trusting it
            if best_col is not None and best_frac >= 0.45:
                logger.info("Detected date column by content: %s (fraction=%0.2f)", best_col, best_frac)
                date_col = best_col

        if date_col is None:
            logger.error("No date column detected in file %s; columns: %s", p, df_read.columns.tolist())
            raise ValueError("No date column detected. Please ensure your files have a date column.")

        # Parse/convert date column, handling excel serials if needed
        series = df_read[date_col]
        parsed = _parse_dates(series, dayfirst=False)
        frac_parsed = parsed.notna().mean() if parsed.size else 0.0
        if frac_parsed < 0.5:
            # maybe numeric serials
            serial_frac = _is_excel_serial_like(series)
            if serial_frac > frac_parsed:
                logger.info("Converting column %s from Excel serials (fraction=%0.2f)", date_col, serial_frac)
                parsed = _convert_excel_serial_to_datetime(series)
            else:
                # try more permissive parsing (dayfirst True)
                parsed_alt = _parse_dates(series, dayfirst=True)
                if parsed_alt.notna().mean() > frac_parsed:
                    parsed = parsed_alt
        df_read[date_col] = parsed
        # rename to standard 'date' if necessary
        if date_col != "date":
            df_read.rename(columns={date_col: "date"}, inplace=True)

        # Normalize amounts: either 'amount' present or debit/credit columns
        if amount_col is None and (debit_col or credit_col):
            # clean debit/credit then compute signed amount = credit - debit
            if debit_col:
                df_read[debit_col] = df_read[debit_col].apply(_clean_amount)
            if credit_col:
                df_read[credit_col] = df_read[credit_col].apply(_clean_amount)
            df_read["amount"] = df_read.get(credit_col, pd.Series([0]*len(df_read))).fillna(0.0).astype(float) - \
                                df_read.get(debit_col, pd.Series([0]*len(df_read))).fillna(0.0).astype(float)
            amount_col = "amount"
        elif amount_col:
            df_read[amount_col] = df_read[amount_col].apply(_clean_amount)
            if amount_col != "amount":
                df_read.rename(columns={amount_col: "amount"}, inplace=True)
                amount_col = "amount"

        # Standardize optional description, payee, running_balance
        desc_col = None
        for i, c in enumerate(cols_lower):
            if any(k in c for k in COMMON_DESC_KEYS):
                desc_col = df_read.columns[i]
                break
        if desc_col and desc_col != "description":
            df_read.rename(columns={desc_col: "description"}, inplace=True)

        payee_col = None
        for i, c in enumerate(cols_lower):
            if any(k in c for k in COMMON_PAYEE_KEYS):
                payee_col = df_read.columns[i]
                break
        if payee_col and payee_col != "payee":
            df_read.rename(columns={payee_col: "payee"}, inplace=True)

        # running balance normalize
        bal_col = None
        for i, c in enumerate(cols_lower):
            if "balance" in c:
                bal_col = df_read.columns[i]
                break
        if bal_col and bal_col != "running_balance":
            df_read.rename(columns={bal_col: "running_balance"}, inplace=True)

        # Drop rows with invalid date
        if "date" in df_read.columns:
            df_read = df_read.dropna(subset=["date"])

        # Ensure amount numeric
        if "amount" in df_read.columns:
            df_read["amount"] = pd.to_numeric(df_read["amount"], errors="coerce")
            df_read = df_read[df_read["amount"].notna()]

        # Coerce running_balance to numeric if present
        if "running_balance" in df_read.columns:
            df_read["running_balance"] = pd.to_numeric(df_read["running_balance"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

        all_dfs.append(df_read)

    if not all_dfs:
        raise ValueError("No valid input files found after processing paths: %s" % paths)

    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)

    # Keep standard columns in order
    final_cols = []
    for c in ["date", "amount", "description", "payee", "running_balance"]:
        if c in df_all.columns:
            final_cols.append(c)
    df_all = df_all[final_cols].copy()
    # final cleanup
    if "amount" in df_all.columns:
        df_all["amount"] = df_all["amount"].astype(float)

    return df_all
