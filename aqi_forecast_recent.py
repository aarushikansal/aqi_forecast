"""
AQI forecasting on RECENT data (CPCB daily AQI bulletins, 2015-2023, via Urban Emissions):
Naive baseline vs ARIMA vs LSTM (1-day-ahead, chronological 80/20 split).

Setup: make a folder called 'bulletins' next to this file and put the per-city CSVs in it
(Panipat_AQIBulletins.csv, Haldia_AQIBulletins.csv, ...). Then run:
    python aqi_forecast_recent.py
Everything is written to the 'results_recent' folder (the old 'results' folder is untouched).
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    tf.keras.utils.set_random_seed(42)
    HAVE_TF = True
except Exception as e:  # ARIMA + baseline still run without TensorFlow
    print("TensorFlow not available, LSTM will be skipped:", e)
    HAVE_TF = False

BULLETIN_DIR = "bulletins"
OUT = "results_recent"
WINDOW = 7          # past 7 days -> next day
TEST_FRAC = 0.2     # latest 20% of days are the test set
MAX_GAP = 21        # fill gaps of up to 21 days (linear); longer gaps are left open and split the data
                    # (filled days are used for training but NOT counted when scoring the test period)
# The models need an unbroken daily series, so each city is cut to ONE continuous stretch.
#   "latest"  -> the most recent stretch that is at least MIN_DAYS long (uses the newest data);
#                falls back to the longest stretch if none is that long
#   "longest" -> simply the longest stretch, wherever it is
STRETCH_MODE = "latest"
MIN_DAYS = 500

# city -> note for the report (proxy / reference cities must be declared as such)
CITIES = {
    "Panipat": "IOCL Panipat refinery (same town)",
    "Haldia": "IOCL Haldia refinery (same town)",
    "Guwahati": "IOCL Guwahati refinery (same city)",
    "Chennai": "CPCL Manali refinery, IOCL subsidiary (same city)",
    "Patna": "Proxy for Barauni refinery (not in source)",
    "Ahmedabad": "Proxy for Vadodara refinery (not in source)",
    "Delhi": "Reference city near Panipat/Mathura (not a refinery site)",
    "Kolkata": "Reference metro near Haldia (not a refinery site)",
}

os.makedirs(OUT, exist_ok=True)


def load_city(city):
    """Read one Urban Emissions bulletin CSV -> (daily AQI series, mean stations reporting)."""
    path = os.path.join(BULLETIN_DIR, f"{city}_AQIBulletins.csv")
    if not os.path.exists(path):
        return None, None
    d = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in d.columns}
    date_col = cols.get("date")
    aqi_col = cols.get("index value") or cols.get("aqi") or cols.get("aqi value")
    if date_col is None or aqi_col is None:
        raise ValueError(f"{path}: expected 'date' and 'Index Value' columns, found {list(d.columns)}")
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[aqi_col] = pd.to_numeric(d[aqi_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    series = d.groupby(date_col)[aqi_col].mean().sort_index()
    st_col = cols.get("no. stations")
    stations = float(pd.to_numeric(d[st_col], errors="coerce").mean()) if st_col else float("nan")
    return series, stations


def find_runs(s):
    """All stretches of consecutive non-missing days, as half-open (start_i, end_i) positions."""
    ok = list(s.notna().values) + [False]
    runs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i))
            start = None
    return runs


def select_stretch(s):
    """Pick the stretch to model (see STRETCH_MODE). Returns (series, description of all long stretches)."""
    runs = find_runs(s)
    if not runs:
        return s.iloc[0:0], ""
    long_runs = [r for r in runs if r[1] - r[0] >= 90]
    desc = "; ".join(f"{s.index[a].date()}->{s.index[b - 1].date()} ({b - a}d)" for a, b in long_runs)
    longest = max(runs, key=lambda r: r[1] - r[0])
    candidates = [r for r in runs if r[1] - r[0] >= MIN_DAYS]
    if STRETCH_MODE == "latest" and candidates:
        chosen = max(candidates, key=lambda r: r[1])
    else:
        chosen = longest
    return s.iloc[chosen[0]:chosen[1]], desc


def metrics(actual, pred):
    return (mean_absolute_error(actual, pred),
            float(np.sqrt(mean_squared_error(actual, pred))))


info_rows, metric_rows = [], []

for city, note in CITIES.items():
    raw, avg_stations = load_city(city)
    if raw is None:
        print(f"[skip] {city}: no file {city}_AQIBulletins.csv in '{BULLETIN_DIR}'")
        continue
    print(f"\n=== {city} ===")

    cal_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    s = raw.reindex(cal_idx)
    n_calendar, n_missing = len(s), int(s.isna().sum())

    # Fill only gaps that are entirely <= MAX_GAP days long. (pandas' own `limit=` would also fill
    # the first MAX_GAP days of a longer gap, which invents data.)
    was_missing = s.isna()
    observed = ~was_missing                       # True where the source really had an AQI value
    gap_len = was_missing.groupby((~was_missing).cumsum()).transform("sum")
    s = s.interpolate(limit_area="inside")
    s[was_missing & (gap_len > MAX_GAP)] = np.nan
    s, stretches = select_stretch(s)
    print(f"Continuous stretches (>=90 days): {stretches or 'none'}")
    if len(s) < 300:
        print(f"[skip] {city}: only {len(s)} usable continuous days")
        continue
    print(f"Using {s.index.min().date()} -> {s.index.max().date()} ({len(s)} days)")

    n_test = int(len(s) * TEST_FRAC)
    train, test = s.iloc[:-n_test], s.iloc[-n_test:]
    scored = observed.loc[test.index]             # score only days with a real (not filled-in) AQI

    # --- Baseline: tomorrow = today
    pred_naive = s.shift(1).loc[test.index]

    # --- ARIMA: d from ADF test, (p, q) from AIC grid, fitted on train only
    d = 0 if adfuller(train)[1] < 0.05 else 1
    best_aic, best_order, best_res = np.inf, None, None
    for p in range(4):
        for q in range(4):
            try:
                res = ARIMA(train, order=(p, d, q)).fit()
                if res.aic < best_aic:
                    best_aic, best_order, best_res = res.aic, (p, d, q), res
            except Exception:
                pass
    # one-step-ahead forecasts on the test period using train-fitted parameters
    pred_arima = best_res.apply(s, refit=False).fittedvalues.loc[test.index]

    # --- LSTM: predicts the DAY-TO-DAY CHANGE, then adds it to yesterday's AQI.
    # (Predicting the level directly made the LSTM settle on a smooth, low curve in volatile
    #  cities such as Panipat. Predicting the change starts it from "tomorrow = today".)
    # Inputs per day: scaled AQI level + scaled change. Scalers use TRAIN data only.
    pred_lstm = None
    if HAVE_TF:
        tf.keras.utils.set_random_seed(42)          # same seed for every city
        levels = s.values.astype(float)
        n_train = len(train)
        delta = np.diff(levels, prepend=levels[0])  # delta[i] = AQI[i] - AQI[i-1]
        lvl_scaler = MinMaxScaler().fit(levels[:n_train].reshape(-1, 1))
        d_std = float(delta[1:n_train].std()) or 1.0
        feat = np.column_stack([lvl_scaler.transform(levels.reshape(-1, 1)).ravel(), delta / d_std])
        X = np.array([feat[i - WINDOW:i] for i in range(WINDOW, len(levels))])   # (N, WINDOW, 2)
        y = (delta / d_std)[WINDOW:]                                              # scaled change
        is_test = s.index[WINDOW:].isin(test.index)
        X_tr, y_tr = X[~is_test], y[~is_test]
        # validation = random 10% of TRAIN windows (not the last slice, which is a different season)
        order = np.random.default_rng(42).permutation(len(X_tr))
        n_val = max(1, len(X_tr) // 10)
        val_i, fit_i = order[:n_val], order[n_val:]
        model = models.Sequential([
            layers.Input(shape=(WINDOW, 2)),
            layers.LSTM(32),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_tr[fit_i], y_tr[fit_i], epochs=100, batch_size=32,
                  validation_data=(X_tr[val_i], y_tr[val_i]), verbose=0,
                  callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        pred_delta = model.predict(X[is_test], verbose=0).ravel() * d_std
        yesterday = levels[WINDOW - 1:-1][is_test]    # AQI of the day before each test day
        pred_lstm = pd.Series(yesterday + pred_delta, index=test.index)

    # --- Metrics
    preds = {"Naive (persistence)": pred_naive, "ARIMA": pred_arima}
    if pred_lstm is not None:
        preds["LSTM"] = pred_lstm
    scores = {}
    for name, p_ in preds.items():
        mae, rmse = metrics(test[scored], p_[scored])
        scores[name] = rmse
        metric_rows.append({"City": city, "Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2)})
        print(f"{name:22s} MAE={mae:7.2f}  RMSE={rmse:7.2f}")
    best_name = min(scores, key=scores.get)

    # --- Plot: actual vs predicted on test period
    plt.figure(figsize=(13, 4.5))
    plt.plot(test.index, test, label="Actual", color="black", lw=1.5)
    for name, p_ in preds.items():
        plt.plot(test.index, p_, label=name, lw=1, alpha=0.85)
    plt.title(f"{city}: actual vs predicted AQI (test period, 1-day ahead)")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{city}_forecast.png", dpi=150)
    plt.close()

    # --- 10 worst days of the best model (you fill in the explanation column)
    err = (test - preds[best_name]).abs()[scored].sort_values(ascending=False).head(10)
    worst = pd.DataFrame({
        "Date": err.index.date,
        "Actual_AQI": test.loc[err.index].round(1).values,
        "Predicted_AQI": preds[best_name].loc[err.index].round(1).values,
        "Abs_error": err.round(1).values,
        "Model": best_name,
        "Why (write 1 line: Diwali? stubble burning? interpolated gap?)": "",
    })
    worst.to_csv(f"{OUT}/{city}_top10_errors.csv", index=False)

    info_rows.append({
        "City": city, "Note": note, "Start": s.index.min().date(), "End": s.index.max().date(),
        "Calendar_days_in_raw_range": n_calendar, "Missing_AQI_days_raw": n_missing,
        "Missing_pct_raw": round(100 * n_missing / n_calendar, 1),
        "Avg_stations_reporting": round(avg_stations, 1),
        "Days_used_after_cleaning": len(s), "Train_days": len(train), "Test_days": len(test), "Test_days_scored": int(scored.sum()),
        "ARIMA_order": best_order, "Best_model_by_RMSE": best_name,
        "Stretch_mode": STRETCH_MODE, "All_stretches_90d_plus": stretches,
    })

if not metric_rows:
    raise SystemExit(f"No city could be processed. Check that the CSVs are inside the '{BULLETIN_DIR}' "
                     f"folder (e.g. {BULLETIN_DIR}/Panipat_AQIBulletins.csv) and read the [skip] messages above.")
pd.DataFrame(info_rows).to_csv(f"{OUT}/dataset_info.csv", index=False)
m = pd.DataFrame(metric_rows)
m.to_csv(f"{OUT}/metrics.csv", index=False)
print("\n=== SUMMARY (RMSE) ===")
print(m.pivot(index="City", columns="Model", values="RMSE"))
print(f"\nDone. RESULTS'{OUT}'")
