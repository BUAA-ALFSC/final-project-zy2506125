import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def weighted_global_mean(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lat", "lon"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--var", type=str, default="tas")
    parser.add_argument("--scenario", type=str, default="ssp245")
    parser.add_argument("--year_start", type=int, default=2080)
    parser.add_argument("--year_end", type=int, default=2100)
    parser.add_argument("--out_dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    truth_ds = xr.open_dataset(
        f"{args.data_dir}/outputs_{args.scenario}.nc",
        engine="netcdf4",
    )

    truth = truth_ds[args.var].mean("member").transpose("time", "lat", "lon")

    if args.var in ["pr", "pr90"]:
        truth = truth * 86400.0

    pred_ds = xr.open_dataset(args.pred_path, engine="netcdf4")
    pred = pred_ds[f"{args.var}_pred"].transpose("time", "lat", "lon")

    truth, pred = xr.align(truth, pred, join="inner")

    truth = truth.sel(time=slice(args.year_start, args.year_end))
    pred = pred.sel(time=slice(args.year_start, args.year_end))

    weights = np.cos(np.deg2rad(truth["lat"]))

    truth_global = weighted_global_mean(truth)
    pred_global = weighted_global_mean(pred)

    truth_mean = truth.mean("time")
    pred_mean = pred.mean("time")

    rmse_spatial = float(
        np.sqrt(
            ((pred_mean - truth_mean) ** 2)
            .weighted(weights)
            .mean(("lat", "lon"))
        ).values
    )
    rmse_global = float(
        np.sqrt(((pred_global - truth_global) ** 2).mean("time")).values
    )

    # ClimateBench normalizes both terms by the absolute global-mean target
    # response over the evaluation period, then weights the global term by 5.
    denom = float(np.abs(weighted_global_mean(truth_mean)).values)
    nrmse_spatial = rmse_spatial / denom if denom != 0 else np.nan
    nrmse_global = rmse_global / denom if denom != 0 else np.nan
    nrmse_total = nrmse_spatial + 5.0 * nrmse_global

    df = pd.DataFrame([{
        "scenario": args.scenario,
        "variable": args.var,
        "year_start": args.year_start,
        "year_end": args.year_end,
        "rmse_spatial": rmse_spatial,
        "rmse_global": rmse_global,
        "nrmse_spatial": nrmse_spatial,
        "nrmse_global": nrmse_global,
        "nrmse_total": nrmse_total,
    }])

    out_csv = f"{args.out_dir}/metrics_official_{args.scenario}_{args.var}_{args.year_start}_{args.year_end}.csv"
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print("Saved metrics:", out_csv)


if __name__ == "__main__":
    main()
