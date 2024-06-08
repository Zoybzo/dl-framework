import sys

from config import Config
from arguments import parse_args


def main(args):
    args = parse_args(args)
    props = []
    config = Config(config_file_list=props)


if __name__ == "__main__":
    main(sys.argv[1:])
