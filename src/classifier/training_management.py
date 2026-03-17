import os
from datetime import datetime
import numpy as np
import json
import torch
from torch import nn
from torch.utils import data

from .bench_manager import TypedTimer


class Trainer:

    def __init__(
            self,

            # The model to be trained
            model: nn.Module,

            # The dataloader to be used for training
            train_dataloader: data.DataLoader,

            # The loss function and optimizer to be used for training
            criterion=nn.CrossEntropyLoss(ignore_index=0),
            optimizer=torch.optim.Adam,

            # How many times the dataset is trained through
            epochs: int = 5,
            # Override the epochs param and only use a fraction of the whole dataset
            dataset_fraction: float = None,

            # Optionally provide a validation dataloader
            val_dataloader: data.DataLoader = None,

            # How many steps between each saved checkpoint
            checkpoint_frequency: int = None,

            lr_scheduler=None,
            device='cpu',

            model_dir=None,
            model_name=None,

            model_outputs_adaptor=lambda x: x,
            model_train_step=lambda model, data: model(data[0]),

            record_per_epoch_training_loss=False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters())
        self.epochs = epochs
        self.dataset_fraction = dataset_fraction
        self.val_dataloader = val_dataloader
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir if model_dir else os.path.join('temp', 'checkpoints')
        os.path.exists(self.model_dir) or os.makedirs(self.model_dir)
        self.model_name = model_name if model_name else model.__class__.__name__
        self.model_outputs_adaptor = model_outputs_adaptor
        self.model_train_step = model_train_step
        self.record_per_epoch_training_loss = record_per_epoch_training_loss

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

        self.timer = TypedTimer(self.model_name)

    def to(self, device):
        self.device = device
        self.model.to(device)

    def train(self):
        self.timer.start("train")

        for epoch in range(1, 1 + self.epochs):
            self._train_step(epoch)
            self._validate_step()
            self._log_step(epoch, train_loss=self.loss["train"][-1], val_loss=self.loss["val"][-1] if len(self.loss["val"]) > 0 else None, tts=self.timer.since("train"))

            if self.lr_scheduler: self.lr_scheduler.step()

            if self.checkpoint_frequency and epoch % self.checkpoint_frequency == 0:
                self._save_checkpoint(epoch)

        if self.checkpoint_frequency and self.epochs % self.checkpoint_frequency != 0:
            self._save_checkpoint(self.epochs)

        self.timer.end("train")

    def _train_step(self, epoch=None):
        self.timer.start("_train_step")

        self.model.train()
        running_loss = []

        self.timer.start("train_dataloader")
        i = 1
        for batch_data in self.train_dataloader:
            self.timer.end("train_dataloader")
            self.timer.start("batch")
            batch_data = [data.to(self.device) for data in batch_data]
            labels = batch_data[-1].to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model_train_step(self.model, batch_data)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
            self.timer.end("batch")

            if self.dataset_fraction and i > self.dataset_fraction: break

            if self.dataset_fraction and self.timer.drag("_train_step", 1):
                self._validate_step()
                self._log_step(epoch, running_loss[-1], tts=self.timer.since("train"))
                self.model.train()

            self.timer.start("train_dataloader")
            i += 1

        epoch_loss = np.mean(running_loss)
        if self.record_per_epoch_training_loss:
            self.loss["epoch.train"] = self.loss.get("epoch.train", []) + [running_loss]
        self.loss["train"].append(epoch_loss)
        self.timer.end("_train_step")

    def _validate_step(self):
        if not self.val_dataloader: return
        self.timer.start("_validate_step")

        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                self.timer.start("val_batch")
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())
                self.timer.end("val_batch")

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
        self.timer.end("_validate_step")

    def _log_step(self, epoch=None, train_loss=None, val_loss=None, tts=None):
        messages = []

        if epoch is not None:
            messages.append(f"Epoch: {epoch:2d}/{self.epochs:2d}")

        if train_loss is None:
            if len(self.loss["train"]) > 0: train_loss = self.loss["train"][-1]

        if train_loss is not None:
            messages.append(f"Train Loss: {train_loss:.2f}")

        if val_loss is None:
            if len(self.loss["val"]) > 0: val_loss = self.loss["val"][-1]

        if val_loss is not None:
            messages.append(f"Val Loss: {val_loss:.2f}")

        if tts is not None:
            messages.append(f"TTS: {tts:.2f}")
            if epoch is not None:
                tpe = tts / epoch
                messages.append(f"ETA: {(self.epochs - epoch) * tpe:.2f}")

        print("    ".join(messages))

    def _save_checkpoint(self, epoch):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"checkpoint_{self.model_name}_{epoch:03d}_{date_str}.pt"
        model_path = os.path.join(self.model_dir, model_path)
        torch.save(self.model, model_path)

    def save_model(self):
        model_path = os.path.join(self.model_dir, f"model_{self.model_name}.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        loss_path = os.path.join(self.model_dir, f"loss_{self.model_name}.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
