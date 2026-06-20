import mindspore.nn as nn
import mindspore.ops as ops


class ClimateCNNLSTM(nn.Cell):
    """
    input:  [batch, time, lat, lon, channels]
    output: [batch, 1, lat, lon]
    """

    def __init__(self, in_channels=4, cnn_channels=20, lstm_hidden=25, lat=96, lon=144):
        super().__init__()

        self.lat = lat
        self.lon = lon
        self.cnn_channels = cnn_channels

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.reduce_mean = ops.ReduceMean(keep_dims=False)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnn_channels,
            kernel_size=3,
            pad_mode="same",
            has_bias=True,
        )
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            has_bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.dense = nn.Dense(lstm_hidden, lat * lon)

    def construct(self, x):
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape

        x = self.transpose(x, (0, 1, 4, 2, 3))
        x = self.reshape(x, (b * t, c, h, w))

        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.reduce_mean(x, (2, 3))
        x = self.reshape(x, (b, t, self.cnn_channels))

        out, _ = self.lstm(x)
        last = out[:, -1, :]

        y = self.dense(last)
        y = self.reshape(y, (b, 1, self.lat, self.lon))

        return y
