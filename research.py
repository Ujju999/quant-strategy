# data manipulation
import polars as pl
from typing import List,Dict,Tuple,Union

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




if __name__ == "__main__":
    set_seed(42)