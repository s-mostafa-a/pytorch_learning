class DeviceDataLoader:
    """usage:
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    device = get_default_device()
    ddl = DeviceDataLoader(some_data_loader, device)
    for xb, yb in ddl:
        ...
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield self.to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

    def to_device(self, data, device):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
