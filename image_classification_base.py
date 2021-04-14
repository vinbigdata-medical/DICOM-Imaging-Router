import torch.nn.functional as F
import torch
import torch.nn as nn

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    # get the accuracy of number preds correctly
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

"""
Let's define the model by extending an ImageClassificationBase class which contains helper methods for training
and validation
"""

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels) # Calculate accuracy
        return {
            'val_loss': loss.detach(), 'val_acc': acc
        }

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() # Combine accuracies
        return {
            'val_loss': epoch_loss.item(),
            'val_acc': epoch_acc.item()
        }

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))



