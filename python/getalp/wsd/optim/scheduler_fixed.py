

class SchedulerFixed(object):

    def __init__(self, fixed_lr: float):
        super().__init__()
        self.fixed_lr = fixed_lr

    def get_learning_rate(self, step: int):
        return self.fixed_lr
