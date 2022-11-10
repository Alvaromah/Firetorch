'''
    Based on Segmentation Models: https://github.com/qubvel/segmentation_models.pytorch
'''
import re
import numpy as np

class MetricsLog():
    def __init__(self, criterion, metrics):
        self.logs = {}
        self.criterion = criterion
        self.metrics = metrics or []
        self.loss_values = []
        self.metrics_values = {self.get_name(metric): [] for metric in self.metrics}

    def update(self, loss, dy, dz):
        # update loss logs
        loss_value = loss.item()
        self.loss_values.append(loss_value)
        loss_logs = {self.get_name(self.criterion): np.mean(self.loss_values)}
        self.logs.update(loss_logs)

        # update metrics logs
        for metric in self.metrics:
            metric_value = metric(dz, dy).cpu().detach().numpy()
            self.metrics_values[self.get_name(metric)].append(metric_value)
        metrics_logs = {k: np.mean(v) for k, v in self.metrics_values.items()}
        self.logs.update(metrics_logs)

        # format logs
        return self._format_logs(self.logs)

    def get_name(self, object):
        if hasattr(object, '_name'): return(x._name)
        name = object.__class__.__name__
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s
