# PyTorch Imports
from torch.optim.lr_scheduler import LambdaLR



# Function: Get the scheduler(s)
def get_scheduler(optimizer, num_training_steps, lr, min_lr, num_warmup_steps=0):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # min_lr / self.lr (aka initial lr) because lambda is multiplied by initial lr (can be thought of as a %)
        return max(
            min_lr / lr, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, -1)
