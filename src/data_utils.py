import numpy as np
import xarray as xr


INPUT_VARS = ["CO2", "CH4", "SO2", "BC"]


def standardize_dims(ds):
    rename = {}
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds


def load_input(path):
    ds = xr.open_dataset(path, engine='netcdf4').compute()
    return standardize_dims(ds)


def load_output(path, var):
    ds = xr.open_dataset(path, engine='netcdf4').compute()
    ds = standardize_dims(ds)

    if "member" in ds[var].dims:
        da = ds[var].mean("member")
    else:
        da = ds[var]

    da = da.transpose("time", "lat", "lon")

    if var in ["pr", "pr90"]:
        da = da * 86400.0

    return da.astype("float32")


def concat_input(hist_path, scen_path):
    hist = load_input(hist_path)
    scen = load_input(scen_path)
    return xr.concat([hist, scen], dim="time")


def concat_output(hist_path, scen_path, var):
    hist = load_output(hist_path, var)
    scen = load_output(scen_path, var)
    return xr.concat([hist, scen], dim="time")


def input_dataset_to_array(ds):
    ds = standardize_dims(ds)

    time = ds["time"].values
    lat = ds["lat"].values
    lon = ds["lon"].values

    arrays = []

    for var in INPUT_VARS:
        da = ds[var]

        if set(da.dims) == {"time"}:
            v = da.values.astype(np.float32)
            v = v[:, None, None]
            v = np.broadcast_to(v, (len(time), len(lat), len(lon)))
        else:
            v = da.transpose("time", "lat", "lon").values.astype(np.float32)

        arrays.append(v)

    x = np.stack(arrays, axis=-1)
    return x.astype(np.float32)


def compute_stats(input_datasets):
    stats = {}

    for var in INPUT_VARS:
        values = []

        for ds in input_datasets:
            ds = standardize_dims(ds)
            values.append(ds[var].values.reshape(-1))

        values = np.concatenate(values).astype(np.float64)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))

        if std == 0 or np.isnan(std):
            std = 1.0

        stats[var] = [mean, std]

    return stats


def normalize_input(ds, stats):
    ds = ds.copy()

    for var in INPUT_VARS:
        mean, std = stats[var]
        ds[var] = (ds[var].dims, ((ds[var].values - mean) / std).astype(np.float32))

    return ds


def make_windows(x, y, times, slider=10, start_index=0):
    xs, ys, out_times = [], [], []

    for i in range(start_index, x.shape[0] - slider + 1):
        target_idx = i + slider - 1
        xs.append(x[i:i + slider])
        ys.append(y[target_idx][None, :, :])
        out_times.append(times[target_idx])

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(out_times),
    )


def load_train_arrays(data_dir, var="tas", slider=10):
    scenarios = ["historical", "ssp126", "ssp370", "ssp585", "hist-GHG", "hist-aer"]

    raw_inputs = []
    raw_outputs = []
    start_indices = []

    hist_input_path = f"{data_dir}/inputs_historical.nc"
    hist_output_path = f"{data_dir}/outputs_historical.nc"

    hist_len = load_input(hist_input_path).sizes["time"]

    for scenario in scenarios:
        print("Loading scenario:", scenario)

        if scenario == "historical":
            x_ds = load_input(hist_input_path)
            y_da = load_output(hist_output_path, var)
            start = 0

        elif scenario.startswith("ssp"):
            x_ds = concat_input(hist_input_path, f"{data_dir}/inputs_{scenario}.nc")
            y_da = concat_output(hist_output_path, f"{data_dir}/outputs_{scenario}.nc", var)
            start = hist_len - slider + 1

        else:
            x_ds = load_input(f"{data_dir}/inputs_{scenario}.nc")
            y_da = load_output(f"{data_dir}/outputs_{scenario}.nc", var)
            start = 0

        raw_inputs.append(x_ds)
        raw_outputs.append(y_da)
        start_indices.append(start)

    stats = compute_stats(raw_inputs)

    X_all, Y_all = [], []

    for x_ds, y_da, start in zip(raw_inputs, raw_outputs, start_indices):
        x_ds = normalize_input(x_ds, stats)
        x = input_dataset_to_array(x_ds)
        y = y_da.values.astype(np.float32)
        times = x_ds["time"].values

        xs, ys, _ = make_windows(x, y, times, slider=slider, start_index=start)

        X_all.append(xs)
        Y_all.append(ys)

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    Y = np.concatenate(Y_all, axis=0).astype(np.float32)

    return X, Y, stats
