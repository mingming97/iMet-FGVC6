import os
import torch


class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader, 
                 criterion, 
                 optimizer,
                 train_cfg, 
                 log_cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.epoch = train_cfg['epoch']
        self.log_dir = log_cfg['log_dir']
        self.print_frequency = log_cfg['print_frequency']
        self.lr_cfg = train_cfg['lr_cfg']
        self.validate_thresh = train_cfg['validate_thresh']

        self.accumulate_batch_size = train_cfg.get('accumulate_batch_size', None)
        self.with_accumulate_batch = self.accumulate_batch_size is not None and self.accumulate_batch_size > 0
        if self.with_accumulate_batch:
            assert self.accumulate_batch_size > self.train_dataloader.batch_size
            assert self.accumulate_batch_size % self.train_dataloader.batch_size == 0
            print('Using accumulate batch. Small batch size: {}. Accumulate batch size: {}'.format(
                self.train_dataloader.batch_size, self.accumulate_batch_size))
            self.update_freq = self.accumulate_batch_size // self.train_dataloader.batch_size
            self.accu_counter = 0
             
        checkpoint = train_cfg['checkpoint']
        if checkpoint is not None:
            assert os.path.exists(checkpoint)
            state = torch.load(checkpoint)
            model.load_state_dict(state['model_params'])
            self.start_epoch = state['epoch'] + 1
            self.best_acc = state['acc']
            print(f'load checkpoint, epoch: {self.start_epoch}, acc: {self.best_acc}')
        else:
            self.start_epoch = 0
            self.best_acc = 0


    def _lr_schedule(self, epoch):
        base_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        lr_step = self.lr_cfg['step']
        if epoch in lr_step:
            for param_group, lr in zip(self.optimizer.param_groups, base_lr):
                param_group['lr'] = lr * self.lr_cfg['gamma']


    def train(self):
        for epoch in range(self.start_epoch, self.epoch):
            #self._lr_schedule(epoch)
            #self._train_one_epoch()
            acc = self._validate()
            print(f'epoch {epoch} validate: score {acc:>2.3f}')
            if self.best_acc < acc:
                self.best_acc = acc
                self._save_checkpoint(epoch, acc)


    def _update_params(self, loss):
        if not self.with_accumulate_batch:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.accu_counter += 1
            loss = loss / self.update_freq
            loss.backward()
            if self.accu_counter == self.update_freq:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accu_counter = 0


    def _train_one_epoch(self):
        self.model.train()
        total_loss, total_sample = 0, 0
        for i, (data, label) in enumerate(self.train_dataloader):
            data = data.cuda()
            label = label.cuda()

            pred = self.model(data)
            loss = self.criterion(pred, label)
            loss_value = loss.item()

            self._update_params(loss)

            if self.print_frequency != 0 and (i + 1) % self.print_frequency == 0:
                print('small iter: {} | loss: {:.6f}'.format(i + 1, loss_value))


    def _validate(self):
        self.model.eval()
        total_predict_positive, total_target_positive, total_true_positive = 0, 0, 0
        with torch.no_grad():
            for data, label in self.val_dataloader:
                data = data.cuda()
                label = label.cuda()

                output = self.model(data)
                output = output.sigmoid()
                max_pred, _ = output.max(dim=1, keepdim=True)
                positive_thresh = max_pred * self.validate_thresh
                predict = (output > positive_thresh).long()
                label = label.type_as(predict)

                total_predict_positive += predict.sum().item()
                total_target_positive += label.sum().item()
                total_true_positive += (label & predict).sum().item()
        p = total_true_positive / total_predict_positive
        r = total_true_positive / total_target_positive
        score = 5 * p * r / (4 * p + r)
        return score

    def _save_checkpoint(self, epoch, acc):
        state = {
            'epoch': epoch,
            'acc': acc,
            'model_params': self.model.state_dict()
        }
        torch.save(state, os.path.join(self.log_dir, 'checkpoint.pth'))
