from argparse import ArgumentParser, Namespace
import safitty
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pl_model import LightningModel
from datasets import get_test_augmentations, Dataset
from metrics import eval_from_scores


def prepare_infer_dataloader(args: Namespace) -> DataLoader:
    transforms = get_test_augmentations(args.image_size)
    df = pd.read_csv(args.infer_df)
    dataset = Dataset(
        df, args.root, transforms, args.face_detector, args.with_labels
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return dataloader


def load_model_from_checkpoint(checkpoints: str, device: str) -> LightningModel:
    model = LightningModel.load_from_checkpoint(checkpoints)
    model.eval()
    model.to(device)
    return model


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    return Namespace(**configs)


def infer_model(
    model: LightningModel,
    dataloader: DataLoader,
    device: str = "cpu",
    verbose: bool = False,
    with_labels: bool = True,
) -> Union[Tuple[float, float, float, float, float], List[float]]:
    scores = []
    targets = torch.Tensor()
    with torch.no_grad():
        if verbose:
            dataloader = tqdm(dataloader)
        for batch in dataloader:
            if with_labels:
                images, labels = batch
                labels = labels.float()
                images = images.to(device)
            else:
                images = batch.to(device)
            cues = model.infer(images)

            for i in range(cues.shape[0]):
                score = 1.0 - cues[i, ...].mean().cpu()
                scores.append(score)
            if with_labels:
                targets = torch.cat([targets, labels])
    if with_labels:
        metrics_, best_thr, acc = eval_from_scores(
            np.array(scores), targets.long().numpy()
        )
        acer, apcer, npcer = metrics_
        if verbose:
            print(f"ACER: {acer}")
            print(f"APCER: {apcer}")
            print(f"NPCER: {npcer}")
            print(f"Best accuracy: {acc}")
            print(f"At threshold: {best_thr}")
        return acer, apcer, npcer, acc, best_thr
    else:
        return scores


if __name__ == "__main__":
    args_ = parse_args()
    model_ = load_model_from_checkpoint(args_.checkpoints, args_.device)

    dataloader_ = prepare_infer_dataloader(args_)

    if args_.with_labels:
        acer_, apcer_, npcer_, acc_, best_thr_ = infer_model(
            model_, dataloader_, args_.device, args_.verbose, True
        )
        with open(args_.out_file, "w") as file:
            file.write(f"acer - {acer_}\n")
            file.write(f"apcer - {apcer_}\n")
            file.write(f"npcer - {npcer_}\n")
            file.write(f"acc - {acc_}\n")
            file.write(f"best_thr - {best_thr_}\n")

    else:
        scores_ = infer_model(model_, dataloader_, args_.device, False, False)
        # if you don't have answers you can write your scores into some file
        with open(args_.out_file, "w") as file:
            file.write("\n".join(list(map(lambda x: str(x), scores_))))
