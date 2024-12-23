import torch
from source import Model
from logging import basicConfig, debug, DEBUG


def main():
    basicConfig(level=DEBUG)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    debug(f"Using {device} device")

    model = Model()


if __name__ == "__main__":
    main()
