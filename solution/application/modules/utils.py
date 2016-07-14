__author__ = 'marko'


import time
import os
import logging
import sys

class Timer(object):
    """
    Use as 'with' construct to measure execution time.
    """
    def __init__(self, label='', verbose=False):
        self.verbose = verbose
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration_secs = self.end - self.start
        if self.verbose:
            logging.debug('%s: elapsed time: %f s' % (self.label, self.duration_secs))

def time_usage(func):
    """
    Use by placing @time_usage anotation above function
    :param func: wrapped function
    :return: wrapped function
    """
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        func(*args, **kwargs)
        end_ts = time.time()
        logging.debug("\n%s: elapsed time: %f\n" % (func.__name__, end_ts - beg_ts))
    return wrapper

def list_files(path):
    """
        List all files in given directory
    :param path:
    :return:
    """
    return [os.path.join(path, img_path) for img_path in os.listdir(path)]

def init_dir(dir_path):
    """
    Create directory if one does not exist previously
    :param dir_path:
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class SingleLevelFilter(logging.Filter):
    """
    Logging filter for allowing single level logging or
    restricting particular level.
    """
    def __init__(self, pass_lvl, reject=False):
        self.accept = lambda record: (record.levelno == pass_lvl) ^ reject

    def filter(self, record):
        return self.accept(record)

def add_logging_property(src=sys.stdout, lvl=logging.INFO):
    """
    Log given level to particular stream
    :param src: stream
    :param lvl: logging level
    :return: None
    """
    handler = logging.StreamHandler(src)
    handler.addFilter(SingleLevelFilter(lvl))

    logging.getLogger().addHandler(handler)

def init_logging(log_path='./app.log', min_lvl=logging.DEBUG):
    """
    Initializes logger in a way that INFO is redirected to stdout,
    ERROR and CRITICAL to stderr. Everything (including INFO, ERR and CRITICAL)
    goes to log file as well.
    """
    logging.basicConfig(filename=log_path, level=min_lvl,
                        format='[%(asctime)s] [%(levelname)s]\t %(message)s',
                        datefmt='%H:%M:%S')

    add_logging_property(sys.stdout, logging.INFO)
    add_logging_property(sys.stdout, logging.DEBUG)
    add_logging_property(sys.stderr, logging.ERROR)
    add_logging_property(sys.stderr, logging.CRITICAL)

init_logging()
