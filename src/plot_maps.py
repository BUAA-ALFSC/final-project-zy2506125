import argparse
from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--var", type=str, default="tas")
    parser.add_argument("--scenario", type=str, default="ssp245")
    parser.add_argument("--years", type=int, nargs="+", default=[2030, 2050, 2100])
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
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

    for year in args.years:
        t = truth.sel(time=year)
        p = pred.sel(time=year)
        e = p - t

        for name, da in [("truth", t), ("pred", p), ("error", e)]:
            plt.figure(figsize=(8, 4))
            da.plot()
            plt.title(f"{args.var} {args.scenario} {year} {name}")
            plt.tight_layout()

            out = f"{args.out_dir}/{args.var}_{args.scenario}_{year}_{name}.png"
            plt.savefig(out, dpi=160)
            plt.close()
            print("Saved:", out)


if __name__ == "__main__":
    main()
