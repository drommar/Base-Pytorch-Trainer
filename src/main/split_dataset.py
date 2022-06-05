from options.option_interface import prepare_split_dataset_args
from engine.splitter import Splitter


def main():
    args = prepare_split_dataset_args()
    my_dataset_splitter = Splitter(args)
    my_dataset_splitter.split_dataset()


if __name__ == '__main__':
    main()
