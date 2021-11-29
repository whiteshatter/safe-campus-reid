from .arguments_test_base import ArgumentsBase


class ArgumentsTest(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTest, self).__init__()

        parser = self.parser
        parser.add_argument('-fg', '--flag', type=str, default='test')

        # model info
        parser.add_argument('-rf', '--restore-file', default="static/checkpoint/final_model.pth(1).tar",
                            help='resume model file', metavar='FILE')
