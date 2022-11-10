'''
    Based on Segmentation Models: https://github.com/qubvel/segmentation_models.pytorch
'''
import numpy as np

class MetricsLog():
    def __init__(self, criterion, metrics):
        self.logs = {}
        self.criterion = criterion
        self.metrics = metrics
        self.loss_values = []
        self.metrics_values = {metric.__name__: [] for metric in self.metrics}

    def update(self, loss, dy, dz):
        # update loss logs
        loss_value = loss.item()
        self.loss_values.add(loss_value)
        loss_logs = {self.criterion.__name__: np.mean(self.loss_values)}
        self.logs.update(loss_logs)
        # update metrics logs
        for metric_fn in self.metrics:
            metric_value = metric_fn(dz, dy).cpu().detach().numpy()
            self.metrics_values[metric_fn.__name__].add(metric_value)
        metrics_logs = {k: np.mean(v) for k, v in self.metrics_values.items()}
        self.logs.update(metrics_logs)
        # format logs
        return self._format_logs(self.logs)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s
