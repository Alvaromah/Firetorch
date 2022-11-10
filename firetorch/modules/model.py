import tqdm
import numpy as np

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model():
    def __init__(self, network, criterion=None, optimizer=None, scheduler=None, metrics=None, device=DEVICE):
        self.device = device
        self.network = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

    def train_epoch(self, datasource):
        self.network.train()
        log = MetricsLog(self.criterion, self.metrics)
        with tqdm.tqdm(datasource) as progress:
            progress.desc = 'train'
            for step, batch in enumerate(progress):
                loss, dy, dz = self._optimize(batch)
                log.update(loss, dy, dz)
                progress.postfix = log.format()
            if self.scheduler:
                self.scheduler.step()
        return log

    def valid_epoch(self, datasource):
        self.network.eval()
        log = MetricsLog(self.criterion, self.metrics)
        with tqdm.tqdm(datasource) as progress:
            progress.desc = 'valid'
            with torch.no_grad():
                for step, batch in enumerate(progress):
                    loss, dy, dz = self._validate(batch)
                    progress.postfix = log.update(loss, dy, dz)

    def predict(self, datasource, as_numpy=True):
        self.network.eval()
        with tqdm.tqdm(datasource) as progress:
            progress.desc = 'predict'
            with torch.no_grad():
                preds = {}
                for step, batch in enumerate(progress):
                    ids, dz = self._predict(batch)
                    z = dz.detach()
                    if as_numpy:
                        z = z.cpu().numpy()
                    for id, p in zip(ids, z):
                        preds[id] = p
                return preds

    def _optimize(self, batch):
        self.optimizer.zero_grad()
        x, y = batch
        dx = self._device(x)
        dy = self._device(y)
        dz = self.network(dx)
        loss = self.criterion(dz, dy)
        loss.backward()
        self.optimizer.step()
        return loss, dy, dz

    def _validate(self, batch):
        x, y = batch
        dx = self._device(x)
        dy = self._device(y)
        dz = self.network(dx)
        loss = self.criterion(dz, dy)
        return loss, dy, dz

    def _predict(self, batch):
        id, x = batch
        dx = self._device(x)
        dz = self.network.predict(dx)
        return id, dz

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

