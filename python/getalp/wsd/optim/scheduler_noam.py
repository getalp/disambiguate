

class SchedulerNoam(object):

    def __init__(self, warmup: int, model_size: int):
        super().__init__()
        self.warmup = warmup
        self.model_size = model_size

    def get_learning_rate(self, step: int):
        step = max(1, step)
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
