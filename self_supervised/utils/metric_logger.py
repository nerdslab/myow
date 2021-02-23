
class MetricLogger:
    r"""Keeps track of training and validation curves, by recording:
        - Last value of train and validation metrics.
        - Train and validation metrics corresponding to maximum or minimum validation metric value.
        - Exponential moving average of train and validation metrics.

    Args:
        smoothing_factor (float, Optional): Smoothing factor used in exponential moving average.
            (default: :obj:`0.4`).
        max (bool, Optional): If :obj:`True`, tracks max value. Otherwise, tracks min value. (default: :obj:`True`).
    """
    def __init__(self, smoothing_factor=0.4, max=True):
        self.smoothing_factor = smoothing_factor
        self.max = max

        # init variables
        # last
        self.train_last = None
        self.val_last = None
        self.test_last = None

        # moving average
        self.train_smooth = None
        self.val_smooth = None
        self.test_smooth = None

        # max
        self.train_minmax = None
        self.val_minmax = None
        self.test_minmax = None
        self.step_minmax = None

    def __repr__(self):
        out = "Last: (Train) %.4f (Val) %.4f\n" % (self.train_last, self.val_last)
        out += "Smooth: (Train) %.4f (Val) %.4f\n" % (self.train_smooth, self.val_smooth)
        out += "Max: (Train) %.4f (Val) %.4f\n" % (self.train_minmax, self.val_minmax)
        return out

    def update(self, train_value, val_value, test_value=0., step=None):
        # last values
        self.train_last = train_value
        self.val_last = val_value
        self.test_last = test_value

        # exponential moving average
        self.train_smooth = self.smoothing_factor * train_value + (1 - self.smoothing_factor) * self.train_smooth \
            if self.train_smooth is not None else train_value
        self.val_smooth = self.smoothing_factor * val_value + (1 - self.smoothing_factor) * self.val_smooth \
            if self.val_smooth is not None else val_value
        self.test_smooth = self.smoothing_factor * test_value + (1 - self.smoothing_factor) * self.test_smooth \
            if self.test_smooth is not None else test_value

        # max/min validation accuracy
        if self.val_minmax is None or (self.max and self.val_minmax < val_value) or \
                (not self.max and self.val_minmax > val_value):
            self.train_minmax = train_value
            self.val_minmax = val_value
            self.test_minmax = test_value
            if step:
                self.step_minmax = step

    def __getattr__(self, item):
        if item not in ['train_min', 'train_max', 'val_min', 'val_max', 'test_min', 'test_max']:
            raise AttributeError
        if self.max and item in ['train_min', 'val_min', 'test_min']:
            raise AttributeError('Tracking maximum values, not minimum.')
        if not self.max and item in ['train_max', 'val_max', 'test_max']:
            raise AttributeError('Tracking minimum values, not maximum.')

        if 'train' in item:
            return self.train_minmax
        elif 'val' in item:
            return self.val_minmax
        elif 'test' in item:
            return self.test_minmax
