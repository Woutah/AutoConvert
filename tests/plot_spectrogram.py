
import matplotlib.pyplot as plt
import argparse
import logging
import numpy as np
# import torch
log = logging.getLogger(__name__)


if __name__ == "__main__":
    spect_melgan = np.load(".\\etc\\p225_003_melgan.npy")
    spect_autovc = np.load(".\\etc\\p225_003_autovc.npy")

    fig, ax = plt.subplots(1, 1)

    plt.tight_layout()
    # ax.imshow(spect_melgan.T)
    plt.imshow(spect_melgan.T)
    plt.tight_layout()
    # ax[0].imshow(spect_autovc.T)
    # plt.imshow(spect_melgan.T)
    plt.show()

    plt.imshow(spect_autovc.T)
    plt.tight_layout()
    plt.show()
    