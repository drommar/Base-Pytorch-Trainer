import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Recoder:
    def __init__(self):
        self.__metrics = {}

    def record(self, name, value):
        if name in self.__metrics.keys():
            if torch.is_tensor(value):
                self.__metrics[name].append(value.item())
            else:
                self.__metrics[name].append(value)
        else:
            if torch.is_tensor(value):
                self.__metrics[name] = [value.item()]
            else:
                self.__metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.__metrics.keys():
            kvs[key] = sum(self.__metrics[key]) / len(self.__metrics[key])
        return kvs

    def clear_metrics(self):
        for key in self.__metrics.keys():
            del self.__metrics[key][:]
            self.__metrics[key] = []


class Logger:
    def __init__(self, args):
        self.__writer = SummaryWriter(os.path.join(args.checkpoints_dir, 'logs'))
        self.__recoder = Recoder()
        self.__checkpoints_dir = args.checkpoints_dir
        self.__gpus = args.gpus

    @staticmethod
    def __tensor2img(tensor):
        # implement according to your data
        return tensor.cpu().detach().numpy()

    def record_scalar(self, name, value):
        self.__recoder.record(name, value)

    def clear_scalar_cache(self):
        self.__recoder.clear_metrics()

    def save_curves(self, epoch):
        kvs = self.__recoder.summary()
        for key in kvs.keys():
            self.__writer.add_scalar(key, kvs[key], epoch)

    def save_imgs(self, names2imgs, epoch):
        for name in names2imgs.keys():
            self.__writer.add_image(name, self.__tensor2img(names2imgs[name]), epoch)

    def print_logs(self, epoch, execution_time):
        print('Epoch {}:{{'.format(epoch))
        kvs = self.__recoder.summary()
        for key in kvs.keys():
            self.__writer.add_scalar(key, kvs[key], epoch)
            print('\t', key + ' = {}'.format(kvs[key]))
        print('\t', 'Execution time(in secs) = {}'.format(execution_time))
        print('}')

    def save_checkpoint(self, epoch, model, optimizer, step=0):
        weights_name = 'weights_{epoch:03d}_{step:06d}.pth'.format(epoch=epoch, step=step)
        checkpoint_name = 'checkpoint_{epoch:03d}_{step:06d}.pth'.format(epoch=epoch, step=step)
        weights_files_dir = os.path.join(self.__checkpoints_dir, 'weights_files')
        if not os.path.exists(weights_files_dir):
            os.system('mkdir -p ' + weights_files_dir)
        checkpoint_files_dir = os.path.join(self.__checkpoints_dir, 'checkpoint_files')
        if not os.path.exists(checkpoint_files_dir):
            os.system('mkdir -p ' + checkpoint_files_dir)
        weights_path = os.path.join(weights_files_dir, weights_name)
        checkpoint_path = os.path.join(checkpoint_files_dir, checkpoint_name)
        if self.__gpus == [0]:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(model.state_dict(), weights_path)
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(model.module.state_dict(), weights_path)
            torch.save(checkpoint, checkpoint_path)
