# data manipulation
import polars as pl
from typing import List,Dict,Tuple,Union,Optional

# ML
import torch
import torch.nn as nn
import torch.optim as optim

# Numerical Computation
import numpy as np
import numpy.typing as npt
from datetime import datetime, timedelta

# visualization
import altair
import matplotlib.pyplot as plt

import random
import re
import itertools
from tqdm import tqdm
from pathlib import Path

SEED = 42

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



## Time Series Aggregation

OHLC_AGGS = [
# Price Statistics
    pl.col("price").first().alias("open"),      # Opening Price
    pl.col("price").max().alias("high"),     # Highest Price
    pl.col("price").min().alias("low"),       # Lowest Price
    pl.col("price").last().alias("close")       # Closing Price
]

def sharpe_annualization_factor(interval:str, trading_days_per_year:int = 365, trading_hours_per_day:int = 24) -> float:

    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("Intervals must be similar to '1d','3h', '15m', '30s")

    value,unit = int(match.group(1)), match.group(2)

    mapping = {
        "d": trading_days_per_year / value,
        "h": trading_days_per_year * (trading_hours_per_day / value),
        "m": trading_days_per_year * (trading_hours_per_day * 60 / value),
        "s": trading_days_per_year * (trading_hours_per_day * 3600 / value)
    }

    period = mapping.get(unit)

    if period is None:
        raise ValueError(f"unsupported unit : {unit}")
    
    return np.sqrt(period)

def get_trade_files(directory :str, symbol:str) -> List[Path]:
    dir_path = Path(directory)
    pattern = f"{symbol}-trades*"
    return sorted(dir_path.glob(pattern))


def load_ohlc_timeseries(symbol:str, time_interval:str):
    return load_timeseries(symbol, time_interval, OHLC_AGGS)

def load_timeseries(symbol:str,time_interval:str,aggs:List[pl.Expr],data_path:Optional[str] = None) -> pl.dataframe:
    if data_path is None:
        data_path = './cache'

    files = get_trade_files(data_path, symbol)

    if not files:
        raise FileNotFoundError(f"No files found for {symbol} in {data_path}")

    ts_list = []
    for file in tqdm(files, desc=f"Loading {symbol}", unit= "file"):
        trades = pl.read_parquet(file)

        if "datetime" not in trades.columns:
            raise ValueError(f" Column 'datetime' is not present in {file.name}")

        trades = trades.with_columns(
            pl.col("datetime").cast(pl.Datetime)).sort("datetime")
        

        ts = trades.group_by_dynamic(
            "datetime",
            every = time_interval,
            offset = '0m'
        ).agg(aggs)

        ts_list.append(ts)

    result = pl.concat(ts_list)
    result = result.sort("datetime").unique(subset = ["datetime"])

    return result
    
def plot_static_timeseries(ts:pl.dataframe, symbol:str, col:str, interval_size:str):
    plt.figure(figsize=(12,6))
    plt.plot(ts["datetime"], ts[col], label = col)
    plt.title(f'{symbol} {interval_size} Bars')
    plt.xlabel('time')
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()

def plot_dyn_timeseries(ts:pl.DataFrame,symbol:str, col:str, time_interval:str):
    return altair.Chart(ts).mark_line(tooltip=True).encode(
        x= "datetime",
        y = col
    ).properties(
        width = 800,
        height = 400,
        title = f"{symbol} {time_interval} {col}"
    ).configure_scale(zero = False).add_selection(
        altair.selection_interval(bind = 'scales', encodings = ['x']),
        altair.selection_interval(bind = 'scales', encodings = ['y'])
    )

def add_lags(df:pl.DataFrame, col:str, max_no_lags:int ,forecast_steps:int) -> pl.DataFrame:
    return df.with_columns([pl.col(col).shift(i * forecast_steps) .alias(f'col_lag_{i}') for i in range(1, max_no_lags+1)])

def plot_distribution(data:pl.DataFrame, col:str, label = None, no_bins = 100):
    return altair.Chart(data).mark_bar().encode(
        altair.X(f'{col}:Q', bin = altair.Bin(maxbins = no_bins)),
        y = 'count()'
    ).properties(
        width = 600,
        height = 400,
        title = f'Distribution of {label if label else col}'
    ).configure_scale(zero = False).add_params(
        altair.selection_interval(bind = 'scales')
    )

if __name__ == "__main__":
    set_seed(42)