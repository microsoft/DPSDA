import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pe.callback.callback import Callback
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.metric_item import FloatListMetricItem
from pe.logging import execution_logger

from .dpimagebench_lib.wrn import WideResNet
from .dpimagebench_lib.resnet import ResNet
from .dpimagebench_lib.resnext import ResNeXt
from .dpimagebench_lib.ema import ExponentialMovingAverage


class DPImageBenchClassifyImages(Callback):
    """The callback that evaluates the classification accuracy of the synthetic data following DPImageBench
    (https://github.com/2019ChenGong/DPImageBench).
    """

    def __init__(
        self,
        model_name,
        test_data,
        val_data,
        batch_size=256,
        num_epochs=50,
        n_splits=1,
        lr=0.01,
        lr_scheduler_step_size=20,
        lr_scheduler_gamma=0.2,
        ema_rate=0.9999,
        **model_params,
    ):
        """Constructor.

        :param model_name: The name of the model to use (wrn, resnet, resnext)
        :type model_name: str
        :param test_data: The test data
        :type test_data: :py:class:`pe.data.Data`
        :param val_data: The validation data
        :type val_data: :py:class:`pe.data.Data`
        :param batch_size: The batch size, defaults to 256
        :type batch_size: int, optional
        :param num_epochs: The number of training epochs, defaults to 50
        :type num_epochs: int, optional
        :param n_splits: The number of splits for gradient accumulation, defaults to 1
        :type n_splits: int, optional
        :param lr: The learning rate, defaults to 0.01
        :type lr: float, optional
        :param lr_scheduler_step_size: The step size for the learning rate scheduler, defaults to 20
        :type lr_scheduler_step_size: int, optional
        :param lr_scheduler_gamma: The gamma for the learning rate scheduler, defaults to 0.2
        :type lr_scheduler_gamma: float, optional
        :param ema_rate: The rate for the exponential moving average, defaults to 0.9999
        :type ema_rate: float, optional
        """
        self._model_name = model_name

        self._test_data = test_data
        self._val_data = val_data
        self._num_classes = len(self._test_data.metadata.label_info)

        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._n_splits = n_splits
        self._lr = lr
        self._lr_scheduler_step_size = lr_scheduler_step_size
        self._lr_scheduler_gamma = lr_scheduler_gamma
        self._ema_rate = ema_rate
        self._model_params = model_params
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._num_channels = self._test_data.data_frame[IMAGE_DATA_COLUMN_NAME].values[0].shape[2]
        self._image_size = self._test_data.data_frame[IMAGE_DATA_COLUMN_NAME].values[0].shape[0]

    def _get_images_and_label_from_data(self, data):
        """Getting images and labels from the data.

        :param data: The data object
        :type data: :py:class:`pe.data.Data`
        :return: The images and labels
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        if data is None:
            return None, None
        else:
            images = np.stack(data.data_frame[IMAGE_DATA_COLUMN_NAME].values)
            images = images.transpose((0, 3, 1, 2)) / 255.0
            labels = np.array(data.data_frame[LABEL_ID_COLUMN_NAME].values)
            return images, labels

    def _get_model(self):
        """Getting the model.

        :raises ValueError: If the model name is unknown
        :return: The model
        :rtype: torch.nn.Module
        """
        if self._model_name == "wrn":
            model = WideResNet(
                in_c=self._num_channels,
                img_size=self._image_size,
                num_classes=self._num_classes,
                depth=28,
                widen_factor=10,
                dropRate=0.3,
                **self._model_params,
            )
        elif self._model_name == "resnet":
            model = ResNet(
                in_c=self._num_channels,
                img_size=self._image_size,
                num_classes=self._num_classes,
                depth=164,
                block_name="BasicBlock",
                **self._model_params,
            )
        elif self._model_name == "resnext":
            model = ResNeXt(
                in_c=self._num_channels,
                img_size=self._image_size,
                cardinality=8,
                depth=28,
                num_classes=self._num_classes,
                widen_factor=10,
                dropRate=0.3,
                **self._model_params,
            )
        else:
            raise ValueError(f"Unknown model name: {self._model_name}")
        return model

    def _get_data_loader(self, data):
        """Getting the data loader.

        :param data: The data object
        :type data: :py:class:`pe.data.Data`
        :return: The data loader
        :rtype: torch.utils.data.DataLoader
        """
        images, labels = self._get_images_and_label_from_data(data)
        if images is None:
            return None
        else:
            return DataLoader(
                TensorDataset(torch.from_numpy(images).float(), torch.from_numpy(labels).long()),
                shuffle=True,
                batch_size=self._batch_size // self._n_splits,
            )

    @torch.no_grad()
    def evaluate(self, model, ema, data_loader, criterion):
        """Evaluating the model.

        :param model: The model
        :type model: torch.nn.Module
        :param ema: The exponential moving average object
        :type ema: :py:class:`pe.callback.image.dpimagebench_lib.ema.ExponentialMovingAverage`
        :param data_loader: The data loader
        :type data_loader: torch.utils.data.DataLoader
        :param criterion: The criterion
        :type criterion: torch.nn.Module
        :return: The accuracy and loss
        :rtype: tuple[float, float]
        """
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

        total = 0
        correct = 0
        loss = 0
        num_batches = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self._device) * 2.0 - 1.0, targets.to(self._device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            num_batches += 1

        ema.restore(model.parameters())

        return correct / total * 100, loss / num_batches

    def __call__(self, syn_data):
        """This function is called after each PE iteration that computes the downstream classification metrics.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The classification accuracy metrics
        :rtype: list[:py:class:`pe.metric_item.FloatListMetricItem`]
        """
        execution_logger.info(f"Evaluating DPImageBench classification accuracy using {self._model_name}")

        model = self._get_model()
        optimizer = optim.Adam(model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self._lr_scheduler_step_size, gamma=self._lr_scheduler_gamma
        )
        criterion = nn.CrossEntropyLoss()

        model = model.to(self._device)
        ema = ExponentialMovingAverage(model.parameters(), self._ema_rate)

        train_loader = self._get_data_loader(syn_data)
        val_loader = self._get_data_loader(self._val_data)
        test_loader = self._get_data_loader(self._test_data)

        grad_accu_step = 0

        train_acc_list = []
        train_loss_list = []

        test_acc_list = []
        test_loss_list = []

        val_acc_list = []
        val_loss_list = []

        for epoch in range(self._num_epochs):
            model.train()
            train_loss = 0
            train_total = 0
            train_correct = 0
            train_num_batches = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device) * 2.0 - 1.0, targets.to(self._device)

                if grad_accu_step == 0:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                (loss / self._n_splits).backward()
                grad_accu_step += 1

                if grad_accu_step == self._n_splits:
                    optimizer.step()
                    grad_accu_step = 0
                    ema.update(model.parameters())

                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                train_num_batches += 1

            scheduler.step()

            train_acc = train_correct / train_total * 100
            train_loss = train_loss / train_num_batches
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

            val_acc, val_loss = self.evaluate(model=model, ema=ema, data_loader=val_loader, criterion=criterion)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

            test_acc, test_loss = self.evaluate(model=model, ema=ema, data_loader=test_loader, criterion=criterion)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

            execution_logger.info(
                f"Epoch {epoch + 1}/{self._num_epochs}, "
                f"Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                f"Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}"
            )

        metric_items = [
            FloatListMetricItem(name=f"{self._model_name}_train_acc", value=train_acc_list),
            FloatListMetricItem(name=f"{self._model_name}_train_loss", value=train_loss_list),
            FloatListMetricItem(name=f"{self._model_name}_val_acc", value=val_acc_list),
            FloatListMetricItem(name=f"{self._model_name}_val_loss", value=val_loss_list),
            FloatListMetricItem(name=f"{self._model_name}_test_acc", value=test_acc_list),
            FloatListMetricItem(name=f"{self._model_name}_test_loss", value=test_loss_list),
        ]

        return metric_items
