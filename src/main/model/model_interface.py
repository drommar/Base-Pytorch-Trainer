from model.base.fcn import base_model
from model.better.fcn import better_model
from model.efficient.fcn import efficient_model


def select_model(args):
    type2model = {
        'base_model': base_model(),
        'better_model': better_model(),
        'efficient_model': efficient_model(),
    }
    model = type2model[args.model_type]
    return model


def main():
    my_net = better_model()
    print(my_net)


if __name__ == '__main__':
    main()
