#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import logging
from datetime import datetime


#---------------------------------------------------------------------------
#   Since we use decorator below to wrap info, debug, error...functions.
#   We need to REWRITE the tracing frame DEPTH to track deeper one layer.
#   This code is just copied from logging.py library but modified depth.
#---------------------------------------------------------------------------
if not hasattr(sys, '_getframe'):
    logging.currentframe = lambda: sys._getframe(5)
else: #pragma: no cover
    def currentframe():
        """Return the frame object for the caller's stack frame."""
        try:
            raise Exception
        except Exception:
            return sys.exc_info()[2].tb_frame.f_back.f_back.f_back.f_back.f_back
    logging.currentframe = currentframe
#---------------------------------------------------------------------------
#   Rewriting ends here
#---------------------------------------------------------------------------


NOTSET = 0
DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4
CRITICAL = 5
LOGOFF = 6

LEVEL = {
    NOTSET: logging.NOTSET,
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARN: logging.WARNING,
    ERROR: logging.ERROR,
    CRITICAL: logging.CRITICAL,
    LOGOFF: 'turn logging off'
}

MODE_FILE = 1
MODE_CONSOLE = 2
MODE_ALL = 3

MODE = {
    MODE_FILE: 'print log in file',
    MODE_CONSOLE: 'print log in console',
    MODE_ALL: 'print log in both file and console'
}

project_name = ''   # use changable type, otherwise can not be modified

def pack_args(func):
    """Decorator to upack args and concatenate to string"""
    def wrapper(*args, **kwargs):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return func(' '.join(map(str, args)), extra={'time': now})
    return wrapper

# these are root logger at default
# users import these variables directly
# it only log after init_log function has been called properly
LOG_DEBUG = pack_args(logging.debug)
LOG_INFO = pack_args(logging.info)
LOG_WARN = pack_args(logging.warn)
LOG_ERROR = pack_args(logging.error)
LOG_CRITICAL = pack_args(logging.critical)

def init_log(name, level, path='', mode=MODE_ALL):
    """
    Init this project logging
    1. choose log mode
    2. rename root logger
    3. set log file if needed
    4. set console log if needed
    --------------------------------
    In: (str, int, str, int)
    --------------------------------
    """
    global project_name         # need to assign this later

    # if level if off or log has been inited, then do nothing
    if level == LOGOFF or project_name:
        return

    # default path is the folder of __main__
    datefmt = '%Y-%m-%d_%H-%M-%S'
    strtime = datetime.now().strftime(datefmt)
    filename = 'Log_{}_{}.txt'.format(name, strtime)

    # if path is not given, use default
    if path and os.path.exists(path):
        filename = os.path.normpath(path) + '/' + filename

    # set logging format
    fmt = '[%(time)s][%(name)s][%(levelname)s][%(process)d] %(message)s (%(filename)s:%(lineno)d:%(funcName)s)'
    formatter = logging.Formatter(fmt=fmt)

    # get root logger
    logger = logging.getLogger()
    logger.setLevel(LEVEL[level])
    logger.name = name

    project_name = name

    # choose to log in file or console or both
    if mode & MODE_FILE:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(LEVEL[level])
        file_handler.setFormatter(formatter) 
        logger.addHandler(file_handler)

    if mode & MODE_CONSOLE:
        stream = logging.StreamHandler()
        stream.setLevel(LEVEL[level])
        stream.setFormatter(formatter)
        logger.addHandler(stream)


def get_name():
    """Get root name, also project name"""
    return project_name

def get_logger():
    """Return wrapped logging methods"""
    return (LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL)
