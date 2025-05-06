import torch

from .bigvgan import load_uninit_bigvgan


def main():
    device = torch.device("cpu")
    bigvgan = load_uninit_bigvgan("config_base_22khz_80band", device)

    with torch.no_grad():
        x = torch.randn(1, bigvgan.h.num_mels, 300).to(device)

        print(device)
        print(x.shape)
        y_g_hat = bigvgan(x)
        print(y_g_hat.shape)


main()
