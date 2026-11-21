# AI usage

## Tools I used, and for what

- Claude / Antigravity for drafting the Population Stability Index (PSI) quantile binning in `nour/drift.py` and configuring Pydantic v2 schemas for `PredictRequest` / `PredictResponse`.
- pytest and custom test harnesses to verify regex boundary conditions across ASCII, Arabic-Indic, and Persian digit scripts.

## The PII cases my AI-written regex missed at first, and how I found them

1. **Whitespace swallowing in phone numbers:** The initial phone regex had an optional separator `[\s-]` positioned immediately after an optional prefix group. When no prefix was present, it matched the preceding whitespace before a standard 11-digit number (e.g., matching `" 01012345678"` inside `"call 01012345678"` and producing `"call[PHONE]"`). I caught this when asserting exact character offsets against bystander text and fixed it by anchoring separators strictly to the presence of a country prefix.
2. **Invalid calendar dates in 14-digit numbers:** The first regex treated any 14-digit number starting with 2 or 3 as a national ID without verifying date validity. I added calendar date validation (`datetime.date(century + yy, mm, dd)`) to reject impossible calendar dates (such as month 13, day 31 in a 30-day month, or February 29 in non-leap years like 1999) as well as unassigned governorate codes.

## One sentence on what my service does when the model is down, written for a pharmacist

If the AI forecasting model is temporarily unavailable or supplier metrics exceed normal ranges, the system instantly calculates your shortage risk using our transparent, rule-based clinical stock formula so you are never left without an immediate, reliable answer at the dispensing counter.
