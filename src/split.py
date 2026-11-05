"""Chronological 70/15/15 split with a 24h purge gap.

Run:  python -m src.split
Out:  data/processed/{train,val,test}.csv, split_info.json
"""
import json
import pandas as pd
from src import config


def chronological_split(df: pd.DataFrame):
    times = pd.Series(sorted(df["datetime_pkt"].unique()))
    n = len(times)
    t1 = pd.Timestamp(times.iloc[int(n * config.TRAIN_FRAC)])
    t2 = pd.Timestamp(times.iloc[int(n * (config.TRAIN_FRAC + config.VAL_FRAC))])
    gap = pd.Timedelta(hours=config.HORIZON_HOURS)

    t = df["datetime_pkt"]
    # A row at time t has its label at t+24h. Drop rows whose label would land in the
    # next split, so no label in train/val is observed during the following period.
    train = df[t < t1 - gap]
    val = df[(t >= t1) & (t < t2 - gap)]
    test = df[t >= t2]
    return train, val, test, t1, t2


if __name__ == "__main__":
    df = pd.read_csv(config.FEATURES_FILE, parse_dates=["datetime_pkt"])
    train, val, test, t1, t2 = chronological_split(df)

    info = {"val_starts": str(t1), "test_starts": str(t2), "purge_gap_hours": config.HORIZON_HOURS}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
        info[name] = {
            "rows": len(part),
            "start": str(part["datetime_pkt"].min()),
            "end": str(part["datetime_pkt"].max()),
            "rows_per_city": part["city"].value_counts().to_dict(),
        }
    config.SPLIT_INFO_FILE.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))