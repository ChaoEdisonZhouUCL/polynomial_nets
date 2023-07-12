import os
from argparse import ArgumentParser
import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from backbones.PiNet import Pinet18
from tqdm import tqdm
from GPU_selector import auto_select_GPU


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main(args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="PolyNet-project",
        # track hyperparameters and run metadata
        config={
            "architecture": "PiNet",
            "dataset": "CIFAR-10",
            "epochs": args.max_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
        },
    )
    # Device configuration
    device = auto_select_GPU()

    #  load  dataset
    #  1. CIFAR10
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
            ),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        # drop_last=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        # drop_last=True,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # load model
    # 1. Pinet18
    model = Pinet18(pretrained=False).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"param": trainable_params})

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=0.9,
    )

    # Train the model
    curr_lr = args.learning_rate
    for epoch in range(args.max_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(tqdm(trainloader)):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # after each epoch, report the performance
        train_loss = train_loss / i
        train_acc = 100 * correct / total
        print(
            "Epoch [{}/{}]] train Loss: {:.4f} train acc: {:.4f}%".format(
                epoch + 1, args.max_epochs, train_loss, train_acc
            )
        )

        # Decay learning rate
        if (epoch + 1) % 20 == 0 and epoch + 1 > 30:
            curr_lr *= 0.1
            update_lr(optimizer, curr_lr)
        wandb.log(
            {
                "loss/train": train_loss,
                "acc/train": train_acc,
                "lr": curr_lr,
            }
        )

        # eval performance
        if (epoch + 1) % 20 == 0:
            # eval
            correct = 0
            total = 0
            with torch.no_grad():
                for i, data in enumerate(tqdm(testloader)):
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # after each epoch, report the performance
            val_loss = val_loss / i
            val_acc = 100 * correct / total
            print(
                "Epoch [{}/{}]] val Loss: {:.4f} val acc: {:.4f}%".format(
                    epoch + 1, args.max_epochs, val_loss, val_acc
                )
            )
            wandb.log({"loss/val": val_loss, "acc/val": val_acc})
    # final performance
    correct = 0
    total = 0
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # after eval, report the performance
        val_loss = val_loss / i
        val_acc = 100 * correct / total
        print(
            "Resnet18 on CIFAR10 test set Loss: {:.4f} acc: {:.4f}%".format(
                val_loss, val_acc
            )
        )
        wandb.log({"loss/test": val_loss, "acc/test": val_acc})
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    # parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    # parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    # parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    # parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    # parser.add_argument(
    #     "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    # )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="Pinet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=120)

    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()

    main(args)
