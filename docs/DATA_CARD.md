# Dataset card

| Property | Value |
| --- | --- |
| File | `main.csv` (original supplied bytes retained) |
| Observations | 2,271 |
| Date window | 2008-01-02 to 2018-05-16 |
| Missing cells / duplicate dates | 0 / 0 |
| Ordering | Ascending unique dates |
| Sampling | Irregular dated observations; no calendar reindexing |
| Target | Recorded GLD value at a future observed row |
| Data provenance | User-supplied project dataset; original provider and acquisition process not documented |
| Data license | Not supplied; no new license or ownership claim is applied to the dataset |

| Field | Interpretation from report | Known limitations |
| --- | --- | --- |
| Date | Observation date | Intraday timestamp, exchange calendar and timezone unknown |
| GLD | Gold-linked price proxy | No verified high/close, adjustment convention, or spot-gold unit metadata |
| SPX | S&P 500 index | Timestamp/adjustment convention unknown |
| USO | US Oil Fund price proxy | Not a direct oil reserve quantity or crude spot quote |
| SLV | Silver-linked price proxy | Not a verified physical silver spot quote |
| EUR/USD | Euro/US dollar exchange rate | Close time unknown |
| EFFR | Effective federal funds rate | Release/vintage timing unknown |
| daily_inflation_rate | Supplied derived inflation field | Formula, publication date, revision vintage and units not documented |

Outliers are retained. Missing dates are not invented. No fresh data is silently downloaded. Dataset SHA-256 is stored in `artifacts/metadata.json`. Treat results as conditional on this supplied snapshot, not as a validated live forecasting feed.

Before reusing the data beyond this project, recover its original source, redistribution terms, field definitions, market timestamps and macro release vintages. No dataset provider or license is guessed from similarly named public datasets.
