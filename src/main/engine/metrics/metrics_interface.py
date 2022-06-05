import torch
from engine.metrics.loss.focal_loss import FocalLoss


def evaluate_accuracy(output_batch: torch.Tensor, label_batch: torch.Tensor) -> float:
    num_acc_label = (output_batch.argmax(dim=1) == label_batch).float().sum().item()
    num_total_label = label_batch.shape[0]
    acc = num_acc_label / num_total_label
    return acc


def calculate_loss(output_batch: torch.Tensor, label_batch: torch.Tensor) -> torch.Tensor:
    loss_function = FocalLoss(num_class=10)
    loss = loss_function(output_batch, label_batch)
    return loss


def main():
    prediction = torch.tensor([[1000., 0, 0, 0], [1000., 0, 0, 0], [1000., 0, 0, 0], [1000., 0, 0, 0]])
    label = torch.tensor([0, 0, 0, 0])
    loss = calculate_loss(prediction, label)
    print(loss)


if __name__ == '__main__':
    main()
