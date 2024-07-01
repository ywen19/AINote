"""
Module for train and validation method definition
"""

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


def get_eval_scores(logits, ground_truth):
    y_pred = logits.argmax(axis=1).cpu().numpy()
    y_true = ground_truth.argmax(axis=1).cpu().numpy()

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


def val_batch(model, criterion, optimizer, batch, device):
    ground_truth = batch['sentiment_type'].float().to(device)
    logits = model(batch)  # logits: [batch_size, classes_num]
    loss = criterion(logits, ground_truth)

    acc, f1 = get_eval_scores(logits, ground_truth)
    return loss.item(), acc, f1


def val_epoch(model, data_loader, criterion, optimizer, device):
    model.eval()

    data_loader.refresh()

    epoch_loss, epoch_acc, epoch_f1 = 0, 0, 0
    batch_amount = len(data_loader)

    while not data_loader.empty():
        batch = next(iter(data_loader))
        batch['context_mask'] = batch['context_mask'].float().to(device)
        batch['context_idxs'] = batch['context_idxs'].to(device)

        loss, acc, f1 = val_batch(model, criterion, optimizer, batch, device)
        del batch
        epoch_loss += loss/batch_amount
        epoch_acc += acc/batch_amount
        epoch_f1 += f1/batch_amount

    return epoch_loss, epoch_acc, epoch_f1


def train_batch(model, criterion, optimizer, batch, device):
    # each batch is a step
    ground_truth = batch['sentiment_type'].float().to(device)
    logits = model(batch)  # logits: [batch_size, classes_num]

    loss = criterion(logits, ground_truth)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    # after completing training on one epoch
    # we will run evaluation on both the train data and dev data
    model.train()
    progress_bar = tqdm(total=len(train_loader))

    # epoch_losses = []

    while not train_loader.empty():
        batch = next(iter(train_loader))
        batch['context_mask'] = batch['context_mask'].float().to(device)
        batch['context_idxs'] = batch['context_idxs'].to(device)

        loss = train_batch(model, criterion, optimizer, batch, device)
        # epoch_losses.append(loss)
        del batch

        progress_bar.update(1)

    # check model performances on train data and test data
    # train_loss, train_acc, train_f1 = val_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_f1 = val_epoch(model, val_loader, criterion, optimizer, device)

    return val_loss, val_acc, val_f1
