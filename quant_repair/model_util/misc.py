def weights_getter(loader, device):
    def get1(key):
        return loader.get(key, dequant=True)[key].to(device)
    return get1
