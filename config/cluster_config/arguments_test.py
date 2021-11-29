from .arguments_test_base import ArgumentsBase


class ArgumentsTest(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTest, self).__init__()

        parser = self.parser
        parser.add_argument('-fg', '--flag', type=str, default='test')

        # model info
        parser.add_argument('-rf', '--restore-file', default="static/checkpoint/cluster/strong-baseline-duke-bnneck-ibn-a-stage1/e159t110799.pth.tar",
                            help='resume model file', metavar='FILE')
