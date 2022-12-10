import torch
from tqdm.notebook import tqdm
from src.utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer):
    """train function for Model

    Args:
        epoch (int): Number of epochs
        data_loader (DataLoader): Torch dataloader
        model (nn.Module): model object
        criterion (function): loss function
        optimizer (function): optimizer

    Returns:
        loss, accuracy: average of loss and accuracy
    """
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    for i, (inputs, targets) in pbar:

        targets = targets.type(torch.cuda.LongTensor)
        inputs = inputs.cuda()

        bs, frames_to_use, *_ = inputs.shape
        outputs = model(inputs)  # call the model with inputs
        results = torch.zeros((bs, 2)).cuda()

        # takes each sequence of frames that composes one video and calculate the mean output as the result
        for k, j in enumerate(range(0, bs * frames_to_use, frames_to_use)):
            results[k] = torch.mean(outputs[j : j + frames_to_use], dim=0)

        # calculate loss and accuracy

        loss = criterion(results, targets)
        acc = calculate_accuracy(results, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add comment to progress bar
        description = f"epoch {epoch} loss: {losses.avg:.4f} {accuracies.avg:.2f}"
        pbar.set_description(description)

    return losses.avg, accuracies.avg


#
def run_inference(model, data_loader, criterion):
    """validate/run inference function for Model

    Args:
        data_loader (DataLoader): Torch dataloader
        model (nn.Module): model object
        criterion (function): loss function
        optimizer (function): optimizer

    Returns:
        true, pred: for confusion matrix
        loss, accuracy: to measure model performance
    """

    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    with torch.no_grad():

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))

        for i, (inputs, targets) in pbar:

            targets = targets.cuda().type(torch.cuda.LongTensor)
            inputs = inputs.cuda()

            bs, frames_to_use, *_ = inputs.shape
            outputs = model(inputs)
            results = torch.zeros((bs, 2)).cuda()

            for k, j in enumerate(range(0, bs * frames_to_use, frames_to_use)):
                results[k] = torch.mean(outputs[j : j + frames_to_use], dim=0)

            loss = torch.mean(criterion(results, targets))
            acc = calculate_accuracy(results, targets)
            _, p = torch.max(results, 1)
            true += (targets).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            description = f"valid. loss: {losses.avg:.4f} acc: {accuracies.avg:.2f}"
            pbar.set_description(description)
        # print("\nAccuracy {}".format(accuracies.avg))

    return true, pred, losses.avg, accuracies.avg


# train function for Efficient_GRU_Model
def train_effnetgru(epoch, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    for i, (inputs, targets) in pbar:

        targets = targets.type(torch.cuda.LongTensor)
        inputs = inputs.cuda()
        _, outputs = model(inputs)

        loss = criterion(outputs, targets.type(torch.cuda.LongTensor))
        acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add comment to progress bar
        description = f"epoch {epoch} loss: {losses.avg:.4f} acc: {accuracies.avg:.2f}"
        pbar.set_description(description)

    return losses.avg, accuracies.avg


# test function for Efficient_GRU_Model
def run_effnetgru(model, data_loader, criterion):
    #   print('Testing')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))

        for i, (inputs, targets) in pbar:

            targets = targets.cuda().type(torch.cuda.FloatTensor)
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            loss = torch.mean(criterion(outputs, targets.type(torch.cuda.LongTensor)))
            acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
            _, p = torch.max(outputs, 1)
            true += (
                (targets.type(torch.cuda.LongTensor))
                .detach()
                .cpu()
                .numpy()
                .reshape(len(targets))
                .tolist()
            )
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            description = f" loss: {losses.avg:.4f} acc: {accuracies.avg:.2f}"
            pbar.set_description(description)
        print("\nAccuracy {}".format(accuracies.avg))
    return true, pred, losses.avg, accuracies.avg
