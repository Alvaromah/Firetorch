import tqdm
import numpy as np

import torch

from .metrics import MetricsLog

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model():
    def __init__(self, network, criterion=None, optimizer=None, scheduler=None, metrics=None, device=DEVICE):
        self.device = device
        self.network = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

    def train_epoch(self, datasource, max_steps=None, activation=None):
        self.network.train()
        metric_log = MetricsLog(self.criterion, self.metrics)
        with tqdm.tqdm(datasource) as progress:
            progress.desc = 'train'
            for step, batch in enumerate(progress):
                loss, dy, dz = self._optimize(batch, activation)
                progress.postfix = metric_log.update(loss, dy, dz)
                if step == max_steps - 1:
                    break
            if self.scheduler:
                self.scheduler.step()
        return metric_log.logs

    def valid_epoch(self, datasource, max_steps=None, activation=None):
        self.network.eval()
        metric_log = MetricsLog(self.criterion, self.metrics)
        with tqdm.tqdm(datasource) as progress:
            progress.desc = 'valid'
            with torch.no_grad():
                for step, batch in enumerate(progress):
                    loss, dy, dz = self._validate(batch, activation)
                    progress.postfix = metric_log.update(loss, dy, dz)
                    if step == max_steps - 1:
                        break
        return metric_log.logs

    def predict(self, datasource, max_steps=None, activation=None):
        self.network.eval()
        with tqdm.tqdm(datasource) as progress:
            progress.desc = 'predict'
            with torch.no_grad():
                preds = []
                for step, batch in enumerate(progress):
                    dz = self._predict(batch, activation)
                    z = dz.detach().cpu().numpy()
                    preds.append(z)
                    if step == max_steps - 1:
                        break
                return np.concatenate(preds, axis=0)

    def _optimize(self, batch, activation):
        self.optimizer.zero_grad()
        x, y = batch
        dx = self._device(x)
        dy = self._device(y)
        dz = self.network(dx)
        if activation:
            dz = activation(dz)
        loss = self.criterion(dz, dy)
        loss.backward()
        self.optimizer.step()
        return loss, dy, dz

    def _validate(self, batch, activation):
        x, y = batch
        dx = self._device(x)
        dy = self._device(y)
        dz = self.network(dx)
        if activation:
            dz = activation(dz)
        loss = self.criterion(dz, dy)
        return loss, dy, dz

    def _predict(self, batch, activation):
        x = batch[:1][0]
        dx = self._device(x)
        dz = self.network(dx)
        if activation:
            dz = activation(dz)
        return dz

    def _device(self, t):
        if isinstance(t, list):
            for n in range(len(t)):
                t[n] = self._device(t[n])              
        else:
            t = t.to(self.device)
        return t

    def load(self, path, device=DEVICE):
        self.network.load(path, map_location=device)

    def save(self, path):
        self.network.save(path)
