import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
import mindspore as ms
from mindspore import Tensor, context

from data_utils import (
    load_input,
    load_output,
    concat_input,
    normalize_input,
    input_dataset_to_array,
    make_windows,
)
from model import ClimateCNNLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--var", type=str, default="tas")
    parser.add_argument("--scenario", type=str, default="ssp245")
    parser.add_argument("--slider", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading stats:", args.stats)
    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    hist_input_path = f"{args.data_dir}/inputs_historical.nc"
    scenario_input_path = f"{args.data_dir}/inputs_{args.scenario}.nc"
    scenario_output_path = f"{args.data_dir}/outputs_{args.scenario}.nc"

    hist_len = load_input(hist_input_path).sizes["time"]

    print("Preparing input windows...")
    x_ds = concat_input(hist_input_path, scenario_input_path)
    x_ds = normalize_input(x_ds, stats)

    x = input_dataset_to_array(x_ds)

    # dummy y is only used by make_windows to align the target index
    dummy_y = np.zeros((x.shape[0], 96, 144), dtype=np.float32)

    start_index = hist_len - args.slider + 1
    X_test, _, pred_times = make_windows(
        x,
        dummy_y,
        x_ds["time"].values,
        slider=args.slider,
        start_index=start_index,
    )

    print("X_test shape:", X_test.shape)
    print("prediction years:", pred_times[0], "to", pred_times[-1])

    truth = load_output(scenario_output_path, args.var)
    lat = truth["lat"].values
    lon = truth["lon"].values

    net = ClimateCNNLSTM(
        in_channels=4,
        cnn_channels=20,
        lstm_hidden=25,
        lat=len(lat),
        lon=len(lon),
    )

    print("Loading checkpoint:", args.ckpt)
    params = ms.load_checkpoint(args.ckpt)
    ms.load_param_into_net(net, params)
    net.set_train(False)

    preds = []

    print("Predicting...")
    for i in range(0, X_test.shape[0], args.batch_size):
        xb = Tensor(X_test[i:i + args.batch_size], ms.float32)
        yb = net(xb).asnumpy()[:, 0, :, :]
        preds.append(yb)

    pred = np.concatenate(preds, axis=0).astype(np.float32)

    da = xr.DataArray(
        pred,
        dims=("time", "lat", "lon"),
        coords={
            "time": pred_times,
            "lat": lat,
            "lon": lon,
        },
        name=f"{args.var}_pred",
    )

    out_path = f"{args.out_dir}/pred_{args.scenario}_{args.var}.nc"
    da.to_dataset().to_netcdf(out_path)

    print("Saved prediction:", out_path)


if __name__ == "__main__":
    main()
