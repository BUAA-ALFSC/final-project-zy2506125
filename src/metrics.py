import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def weighted_global_mean(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lat", "lon"))


def weighted_spatial_rmse(pred, truth):
    weights = np.cos(np.deg2rad(truth["lat"]))
    rmse_t = np.sqrt(((pred - truth) ** 2).weighted(weights).mean(("lat", "lon")))
    return rmse_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--var", type=str, default="tas")
    parser.add_argument("--scenario", type=str, default="ssp245")
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

    # align in case coordinates have tiny differences
    truth, pred = xr.align(truth, pred, join="inner")

    spatial_rmse_t = weighted_spatial_rmse(pred, truth)
    rmse_spatial = float(spatial_rmse_t.mean().values)

    truth_global = weighted_global_mean(truth)
    pred_global = weighted_global_mean(pred)

    rmse_global = float(np.sqrt(((pred_global - truth_global) ** 2).mean()).values)

    weights = np.cos(np.deg2rad(truth["lat"]))

    spatial_denom = float(
        np.sqrt(
            ((truth - truth.weighted(weights).mean(("lat", "lon"))) ** 2)
            .weighted(weights)
            .mean(("time", "lat", "lon"))
        ).values
    )

    global_denom = float(truth_global.std().values)

    nrmse_spatial = rmse_spatial / spatial_denom if spatial_denom != 0 else np.nan
    nrmse_global = rmse_global / global_denom if global_denom != 0 else np.nan
    nrmse_total = 0.5 * (nrmse_spatial + nrmse_global)

    rows = [
        {
            "scenario": args.scenario,
            "variable": args.var,
            "rmse_spatial": rmse_spatial,
            "rmse_global": rmse_global,
            "nrmse_spatial": nrmse_spatial,
            "nrmse_global": nrmse_global,
            "nrmse_total": nrmse_total,
        }
    ]

    df = pd.DataFrame(rows)

    out_csv = f"{args.out_dir}/metrics_{args.scenario}_{args.var}.csv"
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print("Saved metrics:", out_csv)


if __name__ == "__main__":
    main()
