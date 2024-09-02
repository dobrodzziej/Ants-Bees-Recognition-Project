class MyResNet34(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_channels = 3
        out_channels = 64

        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.first_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.first_relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=512, out_features=1)
        self.softmax = nn.Softmax()
        self.residual_units = nn.ModuleList()
        channel_size = 64
        for channel in [channel_size] * 3:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))

        channel_size = 128
        self.residual_units.append(MyResidualUnitChangeDepth(in_channels=channel_size // 2, out_channels=channel_size, second_stride=2))
        for channel in [channel_size] * 3:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))

        channel_size = 256
        self.residual_units.append(MyResidualUnitChangeDepth(in_channels=channel_size // 2, out_channels=channel_size, second_stride=2))
        for channel in [channel_size] * 5:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))

        channel_size = 512
        self.residual_units.append(MyResidualUnitChangeDepth(in_channels=channel_size // 2, out_channels=channel_size, second_stride=2))
        for channel in [channel_size] * 2:
            self.residual_units.append(MyResidualUnit(in_out_channels=channel, second_stride=1))


    def forward(self, x):
        out = self.first_conv(x)
        out = self.first_batch_norm(out)
        out = self.first_relu(out)
        out = self.max_pool(out)
        for idx, unit in enumerate(self.residual_units):
            out = unit(out)
        out = nn.AvgPool2d(kernel_size=out.shape[2])(out)
        out = out.view(out.shape[0], -1)
        # out = torch.mean(out, dim=(2, 3))
        out = self.fc(out)
        return out # Sigmoid will be applied in the BCEWithLogitsLoss loss function