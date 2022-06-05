import argparse
import yaml


class ConfigArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.config_parser = argparse.ArgumentParser()
        self.config_parser.add_argument("-c", "--config_file_path", default=None, metavar="FILE",
                                        help="where to load YAML configuration")
        self.option_names = []
        super(ConfigArgumentParser, self).__init__(*args, **kwargs)

    def add_override_argument(self, *args, **kwargs):
        arg = super().add_argument(*args, **kwargs)
        self.option_names.append(arg.dest)
        return arg

    def parse_args(self, args=None):
        res, remaining_argv = self.config_parser.parse_known_args(args)
        if res.config_file_path is not None:
            with open(res.config_file_path, "r") as f:
                config_vars = yaml.safe_load(f)
            for key in config_vars:
                if key not in self.option_names:
                    self.error(f"unexpected configuration entry: {key}")
            self.set_defaults(**config_vars)

        return super().parse_args(remaining_argv)


def main():
    parser = ConfigArgumentParser()
    parser.add_override_argument('--gpus', nargs='+', type=int, help='numbers of GPU')
    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
