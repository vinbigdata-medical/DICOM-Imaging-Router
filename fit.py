import torch
import torch.nn as nn
from utils.helper import get_lr, evaluate
from utils import constants
from pytorchtools import EarlyStopping



def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    best_val_acc = 0
    # patience = 20
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    # setup custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # setup one-cycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()

        train_losses = []
        val_losses = []
        lrs = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record and update learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader)
        train_losses_mean = torch.stack(train_losses).mean().item()
        result['train_loss'] = train_losses_mean
        result['lrs'] = lrs
        model.epoch_end(epoch, result)

        # EarlyStopping
        # early_stopping(result['val_loss'], model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        
        # finding out the model with best val_acc
        if result['val_acc'] > best_val_acc:
            best_val_acc = result['val_acc']
              
            # save_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': result['val_loss'],
            }, constants.CHECKPOINT_PATH)

        history.append(result)
    return history
