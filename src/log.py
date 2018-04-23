#!/usr/bin/env python
# encoding: utf-8

"""
@organization:            Continental Confidential
@copyright:         Copyright (c) Continental AG. 2018

@license:       This software is furnished under license and may be used or
                copied only in accordance with the terms of such license

@file           log_wrapper.py
@brief          log module, controls the log for the whole project

@author         Chengxing.Zhang@ygomi.com
@date           17/04/2018

@version:       1.0
"""

import os
import sys
import logging
from datetime import datetime


#---------------------------------------------------------------------------
#   Since we use decorator below to wrap info, debug, error...functions.
#   We need to REWRITE the tracing frame DEPTH to track deeper one layer.
#   This code is just copied from logging.py library but modified depth.
#---------------------------------------------------------------------------
if hasattr(sys, '_getframe'):
    logging.currentframe = lambda: sys._getframe(4)
else: #pragma: no cover
    def currentframe():
        """Return the frame object for the caller's stack frame."""
        try:
            raise Exception
        except Exception:
            return sys.exc_info()[2].tb_frame.f_back.f_back.f_back.f_back
    logging.currentframe = currentframe
#---------------------------------------------------------------------------
#   Rewriting ends here
#---------------------------------------------------------------------------


NOTSET = 0
DEBUG = 1
INFO = 2
WARNING = 3
ERROR = 4
CRITICAL = 5

LEVEL = {
    NOTSET: logging.NOTSET,
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARNING: logging.WARNING,
    ERROR: logging.ERROR,
    CRITICAL: logging.CRITICAL
}

MODE_FILE = 1
MODE_CONSOLE = 2
MODE_ALL = 3

MODE = {
    MODE_FILE: 'print log in file',
    MODE_CONSOLE: 'print log in console',
    MODE_ALL: 'print log in both file and console'
}

loggers = {}        # keep track of all loggers
project_name = ''   # use changable type, otherwise can not be modified


def pack_args(func):
    """Decorator to upack args and concatenate to string"""
    def wrapper(*args, **kwargs):
        now = str(datetime.now())[:-3].encode()
        return func(' '.join(map(str, args)), extra={'time':now})
    return wrapper

def init_log(name, level, path='', mode=MODE_ALL):
    """
    Init this project logging
    1. choose log mode
    2. create main logger
    3. set log file if needed
    4. set console log if needed
    5. store logger and project name
    --------------------------------
    In: (str, int, str, int)
    Out: main logger instance
    --------------------------------
    """
    global project_name         # need to assign this later

    # if log has been inited, then return None
    if len(loggers):
        return loggers.get(project_name)

    # if path is not given, use console only as the log place
    if not path:
        path = os.path.dirname(sys.modules['__main__'].__file__)
        datefmt = '%Y-%m-%d_%H-%M-%S'
        strtime = datetime.now().strftime(datefmt)
        filename = 'Log_{}_{}.txt'.format(project_name, strtime)
        
        i = 0
        while os.path.exist(filename):
            i += 1
            filename = filename.replace('.txt', '-{}.txt'.format(i))

    logger = logging.getLogger(name)
    logger.setLevel(LEVEL[level])

    fmt = '[%(time)s][%(name)s][%(levelname)s][%(process)d] %(message)s (%(filename)s:%(lineno)d:%(funcName)s)'
    formatter = logging.Formatter(fmt=fmt)
    
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

    # assign project name, push logger to dictionary
    project_name = name
    loggers[name] = logger

    return (pack_args(logger.debug), pack_args(logger.info),
            pack_args(logger.warning), pack_args(logger.error),
            pack_args(logger.critical))

def create_sub_logger(name):
    """
    Create a sub module logger
    ----------------------------------------
    In: str, name of the sub module
    Out: logger instance from logging module
    ----------------------------------------
    """
    if (not name.startswith(project_name + '.') or
        name in loggers or not len(loggers)):
        return

    sub_logger = logging.getLogger(name)
    loggers[name] = sub_logger
    return (pack_args(sub_logger.debug), pack_args(sub_logger.info),
            pack_args(sub_logger.warning), pack_args(sub_logger.error),
            pack_args(sub_logger.critical))

def get_logger(name=''):
    """
    Get logger instance by name
    if no name given, return project logger.
    if name is not in loggers, return None
    ----------------------------------------
    In: str, name of the module
    Out: logger instance or None
    ----------------------------------------
    """
    if not name:
        name = project_name
    logger = loggers.get(name)
    return (pack_args(logger.debug), pack_args(logger.info),
            pack_args(logger.warning), pack_args(logger.error),
            pack_args(logger.critical))
