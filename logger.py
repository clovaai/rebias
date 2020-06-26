"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import logging


class LoggerBase(object):
    def __init__(self, **kwargs):
        self.level = kwargs.get('level', logging.DEBUG)
        self.logger = self.set_logger(**kwargs)

    def set_logger(self, **kwargs):
        return

    def log(self, msg, level=logging.INFO):
        raise NotImplementedError

    def log_dict(self, msg_dict, prefix='', level=logging.INFO):
        raise NotImplementedError

    def report(self, msg_dict, prefix='', level=logging.INFO):
        raise NotImplementedError


class PrintLogger(LoggerBase):
    def log(self, msg, level=logging.INFO):
        if level <= self.level:
            return

        print(msg)

    def log_dict(self, msg_dict, prefix='Report @step: ', level=logging.INFO):
        if level <= self.level:
            return

        if 'step' in msg_dict:
            step = msg_dict.pop('step')
            print(prefix, step)

        print('{')
        for k, v in sorted(msg_dict.items()):
            print('  {}: {}'.format(k, v))
        print('}')

    def report(self, msg_dict, prefix='Report @step: ', level=logging.INFO):
        self.log_dict(msg_dict, level)


class PythonLogger(LoggerBase):
    def set_logger(self, name=None, level=logging.INFO, fmt=None, datefmt=None):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        if not fmt:
            fmt = '[%(asctime)s] %(message)s'
        if not datefmt:
            datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        return logger

    def log(self, msg, level=logging.INFO):
        self.logger.log(level, msg)

    def log_dict(self, msg_dict, prefix='Report @step: ', level=logging.INFO):
        if 'step' in msg_dict:
            step = msg_dict.pop('step')
            prefix = '{}{:.2f} '.format(prefix, step)
        self.log('{}{}'.format(prefix, msg_dict))

    def report(self, msg_dict, prefix='Report @step', level=logging.INFO):
        self.log_dict(msg_dict, prefix, level)
