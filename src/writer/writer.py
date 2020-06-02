# This class hadles logging functions.
# Usage:
# from writer import Logger
# logger = Logger('train', 'val', filename='log_path.log')
#
# logger.train.info('training loss: %.3f' % loss)
# logger.train.wav(wav_array, 'save_path.wav')
# logger.val.info('starting validation...')

# and the output log is looking like:
# [Time] train  - training loss: 0.344
# [Time] train  - save wav file at <save_path.wav>
# [Time] val    - Starting validation...

import os
import logging

from tensorboardX import SummaryWriter

from .log_util import CustomLogClass


class Logger(object):
    ''' Logger class for the machine learning.

    Args:
        *args: name of the logger instances

    Return:
        Logging class

    '''
    def __init__(self, logname, *args, **base_config):
        if not os.path.exists('./log/log'):
            os.makedirs('./log/log')
        if not os.path.exists('./log/tbx'):
            os.makedirs('./log/tbx')

        if not logname == '':
            logging.basicConfig(filename='./log/log/%s' % logname,
                            format='[%(asctime)s] - %(name)s - %(message)s',
                            level=logging.INFO)
        else:
            logging.basicConfig(format='[%(asctime)s] - %(name)s - %(message)s',
                            level=logging.INFO)
        self.writer = SummaryWriter(logdir='./log/tbx/%s' % logname)

        for name in args:
            child_logger = logging.getLogger(name)
            ch = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s')
            ch.setFormatter(formatter)
            child_logger.addHandler(ch)
            setattr(self, name, CustomLogClass(name, child_logger, self.writer))

    def close(self):
        self.writer.close()
