from argparse import ArgumentParser, Namespace

import safitty
import pytorch_lightning as pl

from pl_model import LightningModel


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    configs = Namespace(**configs)

    model = LightningModel(hparams=configs)
    trainer = pl.Trainer.from_argparse_args(
        configs,
        fast_dev_run=False,
        early_stop_callback=True,
        default_root_dir=configs.default_root_dir,
    )
    trainer.fit(model)
