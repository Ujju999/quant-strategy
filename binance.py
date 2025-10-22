import polars as pl
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

import research

MAKER_FEE = 0.000450
TAKER_FEE = 0.000450

def download_and_unzip(symbol:str, date:str | datetime, download_dir:str = "data", cache_dir:str = "cache") -> pl.dataframe:

    date_str = date.strftime('%Y-%m-%d') if isinstance(date,datetime) else date

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{symbol}-trades-{date_str}.parquet"

    if cache_path.exists():
        return pl.read_parquet(cache_path)
    
    url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"

    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True)
    zip_path = download_dir / f"{symbol}-trades-{date_str}.zip"

    response = requests.get(url, stream = True)
    response.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(download_dir)

    csv_path = download_dir / f"{symbol}-trades-{date_str}.csv"

    df = pl.read_csv(
                    csv_path,
                    schema = {
                        "id":pl.Int64,
                        "price":pl.Float64,
                        "qty":pl.Float64,
                        "quoteQty":pl.Float64,
                        "time":pl.Int64,
                        "isBuyerMaker":pl.Boolean
                    }
                ).with_columns(
                    pl.from_epoch("time", time_unit="ms").alias("datetime")
                )
    df.write_parquet(cache_path)
    zip_path.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)

    return df

def download_trades(symbol:str, num_days:int, download_dir:str = "data", cache_dir:str =  "cache", return_trades = False) -> pl.dataframe:

    yesterday = datetime.now() -timedelta(days = 1)
    start_date = yesterday - timedelta(days = num_days - 1)

    dfs = []
    for i in tqdm(range(num_days), desc= f"downloading {symbol}"):
        current_date = start_date + timedelta(days= i)
        try:
            if return_trades:
                dfs.append(download_and_unzip(symbol, current_date,download_dir,cache_dir))
            else:
                download_and_unzip(symbol, current_date,download_dir,cache_dir)
        except Exception as e:
            tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")
    return pl.concat(dfs) if return_trades else None
