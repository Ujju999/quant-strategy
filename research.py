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
    return df.with_columns([pl.col(col).shift(i * forecast_steps) .alias(f'{col}_lag_{i}') for i in range(1, max_no_lags+1)])

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

def to_tensor(x,dtype = None) -> torch.Tensor:
    return torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)

def timeseries_train_test_split(df:pl.DataFrame,features:List[str],target:str, test_size = 0.25) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    df = df.drop_nulls()
    X = to_tensor(df[features])
    y = to_tensor(df[target]).reshape(-1,1)
    X_train, X_test = timeseries_split(X, test_size)
    y_train, y_test = timeseries_split(y, test_size)
    return X_train,X_test,y_train,y_test

def timeseries_split(t, test_size:float= 0.25):
    if not (0<test_size<1):
        raise ValueError(f"test_size must be between 0 and 1 (got {test_size})")
    split_idx = int(len(t) * (1-test_size))
    return t[:split_idx],t[split_idx:]

def plot_column(df:pl.DataFrame, col_name:str,figsize = (15,6) , title:str = None,xlabel:str = 'Index'):
    if title is None:
        title = col_name
    chart = df[col_name].plot.line()

    return chart.properties(
        width = 800,
        height = 400,
        title = title
    )

def model_trade_results(y_actual:pl.Series,y_pred:pl.Series) ->pl.DataFrame:
    trade_results = pl.DataFrame({
    'y_pred': y_pred.squeeze(),
    'y_true': y_actual.squeeze()
                }).with_columns(
                    (pl.col('y_pred').sign() == pl.col('y_true').sign()).alias('is_win'),
                    pl.col('y_pred').sign().alias('signal')
                ).with_columns(
                    (pl.col('signal') * pl.col('y_true')).alias('trade_log_return')
                ).with_columns(
                    pl.col('trade_log_return').cum_sum().alias('equity_curve')
                ).with_columns(
                    (pl.col('equity_curve') - pl.col('equity_curve').cum_max()).alias('drawdown_log')
                )
    return trade_results

def eval_model_performance(y_actual:pl.Series,y_pred:pl.Series,feature_name:List[str],target_name:str, annualized_rate:float) -> Dict[str, any]:
    trade_results = model_trade_results(y_actual,y_pred)

    accuracy = trade_results['is_win'].mean()
    avg_win = trade_results.filter(pl.col('is_win') == True)['trade_log_return'].mean()
    avg_loss = trade_results.filter(pl.col('is_win') == False)['trade_log_return'].mean()
    expected_value = accuracy * avg_win + (1- accuracy) * avg_loss
    drawdown = (trade_results["equity_curve"] - trade_results["equity_curve"].cum_max())
    max_drawdown = drawdown.min()
    sharpe = trade_results["trade_log_return"].mean() / trade_results["trade_log_return"].std() if trade_results["trade_log_return"].std() > 0 else 0
    annualized_sharpe = sharpe * annualized_rate
    equity_trough = trade_results["equity_curve"].min()
    equity_peak = trade_results["equity_curve"].max()
    std = trade_results["equity_curve"].std()
    total_log_return = trade_results["trade_log_return"].sum()
    return{
        'features': ','.join(list(feature_name)),
        'target':target_name,
        'num_trades' : len(trade_results),
        'win_rate':accuracy,
        'avg_loss':avg_loss,
        'avg_win':avg_win,
        'best_trade':trade_results['trade_log_return'].max(),
        'worst_trade':trade_results["trade_log_return"].min(),
        'ev':expected_value,
        'std':std,
        'total_log_return':total_log_return,
        'compound_return':np.exp(total_log_return),
        'max_drawdown':max_drawdown,
        'equity_trough':equity_trough,
        'equity_peak':equity_peak,
        'sharpe':annualized_sharpe
    }



if __name__ == "__main__":
    set_seed(42)