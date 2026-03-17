import torch

from classifier import *
import os


class SoftCrossEntropyLoss(nn.Module):
    def forward(self, logits, targets):
        targets = targets / targets.sum(dim=1, keepdim=True)
        log_probs = torch.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


def test_training():
    print()

    print("cwd", os.getcwd())

    model = SimpleCNN(dtype=torch.float64)
    print(model)
    print(measure_size(model))

    dir_dataset = OIDv6Dataset(hard_limit=10, root='', dtype=np.float64)
    pipe_dataset = PipelinedDataset(dir_dataset, process_sample)
    dataloader = DataLoader(pipe_dataset, batch_size=16)

    print(next(iter(dataloader))[0].dtype)

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        # criterion = nn.CrossEntropyLoss(ignore_index=0),
        criterion = SoftCrossEntropyLoss(),
        epochs=2,
    )

    trainer.train()

    print("Yahoo")
    print(trainer.loss)


