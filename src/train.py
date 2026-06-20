import argparse
import json
import os
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, context, Tensor
from tqdm import tqdm

from data_utils import load_train_arrays
from model import ClimateCNNLSTM


class NetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, x, y):
        pred = self.backbone(x)
        loss = self.loss_fn(pred, y)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--var", type=str, default="tas")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--slider", type=int, default=10)
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "CPU", "Ascend"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/checkpoints")
    args = parser.parse_args()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading training arrays...")
    X, Y, stats = load_train_arrays(args.data_dir, var=args.var, slider=args.slider)

    if args.max_samples and args.max_samples > 0:
        X = X[:args.max_samples]
        Y = Y[:args.max_samples]

    print("X shape:", X.shape, X.dtype)
    print("Y shape:", Y.shape, Y.dtype)

    train_ds = ds.NumpySlicesDataset(
        {"x": X, "y": Y},
        shuffle=True,
    ).batch(args.batch_size, drop_remainder=False)

    net = ClimateCNNLSTM(
        in_channels=4,
        cnn_channels=20,
        lstm_hidden=25,
        lat=Y.shape[-2],
        lon=Y.shape[-1],
    )

    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr)

    net_with_loss = NetWithLoss(net, loss_fn)
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_net.set_train()

    log_path = os.path.join(args.out_dir, f"train_log_{args.var}.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss\n")

    for epoch in range(1, args.epochs + 1):
        losses = []

        for batch in tqdm(train_ds.create_dict_iterator(), desc=f"Epoch {epoch}/{args.epochs}"):
            x = batch["x"]
            y = batch["y"]
            loss = train_net(x, y)
            losses.append(float(loss.asnumpy()))

        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch}: loss={mean_loss:.6f}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{mean_loss}\n")

        ms.save_checkpoint(net, os.path.join(args.out_dir, f"cnn_lstm_{args.var}_epoch{epoch}.ckpt"))

    final_ckpt = os.path.join(args.out_dir, f"cnn_lstm_{args.var}_final.ckpt")
    ms.save_checkpoint(net, final_ckpt)

    stats_path = os.path.join(args.out_dir, f"stats_{args.var}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Training finished.")
    print("Final checkpoint:", final_ckpt)
    print("Stats:", stats_path)


if __name__ == "__main__":
    main()
