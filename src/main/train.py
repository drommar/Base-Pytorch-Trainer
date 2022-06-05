from options.option_interface import prepare_train_args
from model.model_interface import select_model
from data.data_interface import select_train_loader
from data.data_interface import select_eval_loader
from engine.trainer import Trainer


def main():
    args = prepare_train_args()
    model = select_model(args)
    train_loader = select_train_loader(args)
    eval_loader = select_eval_loader(args)
    my_trainer = Trainer(args, model, train_loader, eval_loader)
    my_trainer.train()


if __name__ == '__main__':
    main()
