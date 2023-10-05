import torch
class EarlyStopping():
    def __init__(self, patience=3, min_delta=0.001, checkpoint_path='checkpoint.pt'):

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.stop = False
        self.wait = 0
        self.path = checkpoint_path

    def __call__(self, current_loss, model):

        loss = -current_loss

        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss < self.best_loss + self.min_delta:
            self.wait += 1
            print(f'EarlyStopping counter: {self.wait} out of {self.patience}')
            if self.wait >= self.patience:
                self.stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(model)
            self.wait = 0

    def save_checkpoint(self, model):
        print(f'Validation loss decreased. Saving model...')
        torch.save(model, self.path)