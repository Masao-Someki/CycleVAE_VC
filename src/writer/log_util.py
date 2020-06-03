# custom log utils class

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)

class CustomLogClass(object):
    def __init__(self, name, logger, writer):
        self.name = name
        self.logger = logger
        self.writer = writer

    def info(self, text):
        self.logger.info(text)

    def figure(self, phase, dic, iter_count):
        # function to log figures like loss.
        for k,v in dic.items():
            self.writer.add_scalar('data/%s/%s' % (phase, k), v, iter_count)
        self.writer.flush()
