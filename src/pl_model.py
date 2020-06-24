import pandas as pd
import numpy as np

from sklearn import metrics

import torch
from torch.nn import functional as F

from catalyst.data.sampler import BalanceClassSampler
import pytorch_lightning as pl

from datasets import (
    Dataset,
    get_test_augmentations,
    get_train_augmentations
)
from models.scan import SCAN
from loss import TripletLoss


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = SCAN()
        self.triplet_loss = TripletLoss()

    def forward(self, x):
        return self.model(x)

    def calc_losses(self, outs, clf_out, target):

        clf_loss = \
            F.cross_entropy(clf_out, target) * \
            self.hparams.loss_coef['clf_loss']
        cue = outs[-1]
        cue = target.reshape(-1, 1, 1, 1) * cue
        num_reg = \
            (torch.sum(target) *
             cue.shape[1] *
             cue.shape[2] *
             cue.shape[3]).type(torch.float)
        reg_loss = \
            (torch.sum(torch.abs(cue)) / (num_reg + 1e-9)) * \
            self.hparams.loss_coef['reg_loss']

        trip_loss = 0
        bs = outs[-1].shape[0]
        for feat in outs[:-1]:
            feat = F.adaptive_avg_pool2d(feat, [1, 1]).view(bs, -1)
            trip_loss += \
                self.triplet_loss(feat, target) \
                * self.hparams.loss_coef['trip_loss']
        total_loss = clf_loss + reg_loss + trip_loss

        return total_loss

    def training_step(self, batch, batch_idx):
        input_ = batch[0]
        target = batch[1]
        outs, clf_out = self(input_)
        loss = self.calc_losses(outs, clf_out, target)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(
            self,
            outputs
    ):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {
            'train_avg_loss': avg_loss,
        }
        return {'train_avg_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ = batch[0]
        target = batch[1]
        outs, clf_out = self(input_)
        loss = self.calc_losses(outs, clf_out, target)
        val_dict = {
            'val_loss': loss,
            'score': clf_out.cpu().numpy(),
            'target': target.cpu().numpy()
        }
        return val_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        targets = np.hstack([output['target'] for output in outputs])
        scores = np.vstack([output['score'] for output in outputs])[:, 1]
        roc_auc = metrics.roc_auc_score(targets, scores)
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_roc_auc': roc_auc
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            milestones=self.hparams.milestones,
            gamma=self.hparams.gamma
        )
        return [optim], [scheduler]

    def train_dataloader(self):
        transforms = get_train_augmentations()
        df = pd.read_csv(self.hparams.train_df)
        dataset = Dataset(df, self.hparams.path_root, transforms)
        labels = list(df.target.values)
        sampler = BalanceClassSampler(
            labels,
            mode='upsampling')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_train,
            sampler=sampler
        )
        return dataloader

    def val_dataloader(self):
        transforms = get_test_augmentations()
        df = pd.read_csv(self.hparams.val_df)
        dataset = Dataset(df, self.hparams.path_root, transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_val,
            shuffle=False
        )
        return dataloader
