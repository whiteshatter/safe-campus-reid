from .arguments_second_base import ArgumentsBase


class ArgumentsTrainVal2(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTrainVal2, self).__init__()

        parser = self.parser
        parser.add_argument('-fg', '--flag', type=str, default='train')

        # log info
        parser.add_argument('-logf', '--log-freq', type=int,
                            default=100, help='log frequency over iterations')

        # dataset loading info
        parser.add_argument('-vb', '--validation-batch-size',
                            type=int, default=16, help='validation batch size')

        # model save info
        parser.add_argument('-cptf', '--checkpoint-freq', default=40, type=int,
                            help='the frequency of saving model over epoches')

        # training control parameters
        parser.add_argument('-lr', '--learning-rate', default=3.5e-6, type=float, help='learning rate')
        parser.add_argument('-me', '--maxepoch', default=160,
                            type=int, help='the number of epochs to train')
        parser.add_argument('-re', '--resume-epoch', type=int,
                            default=119, help='the epoch model resume from')
        parser.add_argument('-ri', '--resume-iteration', type=int,
                            default=27118, help='the iteration model resume from')

        # model info
        parser.add_argument('-rf', '--restore-file', default='E:/gi/safe-campus-reid/static/checkpoint/cluster/e119t231758.pth.tar',
                            help='resume model file', metavar='FILE')

        # log files info
        parser.add_argument('-lcd', '--check-log-dir', metavar='DIR', default='strong-baseline-duke-bnneck-ibn-a-stage2',
                            help='checkpoints and logs directory')
