from options.option_interface import prepare_eval_args
from model.model_interface import select_model
from data.augment import transform_eval
from engine.predicter import Predicter


def main():
    args = prepare_eval_args()
    model = select_model(args)
    my_predicter = Predicter(args, model, transform_eval)
    my_predicter.predict_csv()


if __name__ == '__main__':
    main()
