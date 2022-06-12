import time
import torch
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from engine.utils.logger import Logger
from engine.metrics.metrics_interface import evaluate_accuracy
from engine.metrics.metrics_interface import calculate_loss


class Trainer(object):
    def __init__(self, args, model, train_loader, val_loader):
        torch.manual_seed(args.seed)
        self.__args = args
        self.__logger = Logger(args)
        self.__train_loader = train_loader
        self.__val_loader = val_loader
        self.__start_epoch = 0

        train_status = 'Normal'
        train_status_logs = []

        # loading model
        self.__model = model
        if args.is_load_pretrained_weight:
            train_status = 'Continuance'
            self.__model.load_state_dict(torch.load(args.pretrained_weights_path), strict=args.is_load_strict)
            train_status_logs.append('Log   Output: Loaded pretrained weights successfully')

        if args.is_resuming_training:
            train_status = 'Restoration'
            checkpoint = torch.load(args.checkpoint_path)
            self.__start_epoch = checkpoint['epoch'] + 1
            self.__model.load_state_dict(checkpoint['model_state_dict'], strict=args.is_load_strict)
            train_status_logs.append('Log   Output: Resumed previous model state successfully')

        if args.gpus == [0]:
            gpu_status = 'Single-GPU'
            device = torch.device("cuda:0")
            self.__model.to(device)
        else:
            gpu_status = 'Multi-GPU'
            self.__model = torch.nn.DataParallel(self.__model, device_ids=args.gpus, output_device=args.gpus[0])

        # initialize the optimizer
        self.__optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.__model.parameters()),
                                            self.__args.lr,
                                            betas=(self.__args.momentum, self.__args.beta),
                                            weight_decay=self.__args.weight_decay)
        if args.is_resuming_training:
            checkpoint = torch.load(args.checkpoint_path)
            self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_status_logs.append('Log   Output: Resumed previous optimizer state successfully')

        # print status
        print('****************************************************************************************************')
        print('Model:')
        print(self.__model)
        print('****************************************************************************************************')
        print('Params To Learn:')
        for name, param in self.__model.named_parameters():
            if param.requires_grad:
                print('\t', name)
        print('****************************************************************************************************')
        print('Train Status: ' + train_status)
        print('GPU   Status: ' + gpu_status)
        for train_status_log in train_status_logs:
            print(train_status_log)
        print('****************************************************************************************************')

    def train(self):
        for epoch in range(self.__start_epoch, self.__args.epochs):
            # train for one epoch
            since = time.time()
            self.__train_per_epoch()
            self.__val_per_epoch()
            self.__logger.save_curves(epoch)
            self.__logger.save_checkpoint(epoch, self.__model, self.__optimizer)
            self.__logger.print_logs(epoch, time.time() - since)
            self.__logger.clear_scalar_cache()

    def __train_per_epoch(self):
        # switch to train mode
        self.__model.train()

        for i, data_batch in enumerate(self.__train_loader):
            input_batch, output_batch, label_batch = self.__step(data_batch)

            # compute loss and acc
            loss, metrics = self.__compute_metrics(output_batch, label_batch, is_train=True)

            # compute gradient and do Adam step
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            # logger record
            for key in metrics.keys():
                self.__logger.record_scalar(key, metrics[key])

    def __val_per_epoch(self):
        # switch to eval mode
        self.__model.eval()

        with torch.no_grad():
            for i, data_batch in enumerate(self.__val_loader):
                input_batch, output_batch, label_batch = self.__step(data_batch)

                # compute loss and acc
                loss, metrics = self.__compute_metrics(output_batch, label_batch, is_train=False)

                for key in metrics.keys():
                    self.__logger.record_scalar(key, metrics[key])

    def __step(self, data_batch):
        input_batch, label_batch = data_batch
        # warp input
        input_batch = Variable(input_batch).cuda()
        label_batch = Variable(label_batch).cuda()

        # compute output
        output_batch = self.__model(input_batch)
        return input_batch, output_batch, label_batch

    @staticmethod
    def __compute_metrics(output_batch, label_batch, is_train):
        # you can call functions in metrics_interface.py
        loss = calculate_loss(output_batch, label_batch)
        acc = evaluate_accuracy(output_batch, label_batch)
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'loss': loss.item(),
            prefix + 'accuracy': acc,
        }
        return loss, metrics

    @staticmethod
    def __gen_imgs_to_write(img, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'img': img[0],
        }


def main():
    pass


if __name__ == '__main__':
    main()
