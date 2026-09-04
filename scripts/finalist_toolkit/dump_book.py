"""One-time dump of the REAL after-tax stock-book monthly return series (+SPY)
so the options sweep can measure beta/blend against the actual portfolio."""
import pickle, pandas as pd
from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from data.db import get_db
from backtesting.data_loader import SPY

OUT="/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/options_calibration/book_returns.csv"

def main():
    panel, mr, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
    r_book = pd.Series(fs.config_nav(dates, wts, matrix), index=dates).pct_change().dropna()
    spy_at = pd.Series(fs.bench_nav(matrix, dates, SPY), index=dates).pct_change().dropna()
    df = pd.DataFrame({"book": r_book, "spy_aftertax": spy_at})
    df.index.name = "date"
    df.to_csv(OUT)
    print(f"wrote {len(df)} rows {df.index.min()}..{df.index.max()} -> {OUT}")

if __name__=="__main__":
    main()
