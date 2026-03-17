import numpy as np
import random
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(random.randint(1, 10000))


class seed_as:
    def __init__(self, seed: int = random.randint(1, 10000)):
        self.seed = seed

    def __enter__(self):
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        set_seed(self.seed)

    def __exit__(self, *_):
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        if self.cuda_states is not None: torch.cuda.set_rng_state_all(self.cuda_states)
