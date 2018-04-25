from src.log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
from src import log

log.init_log('test_slam', log.DEBUG, mode=log.MODE_CONSOLE)

LOG_DEBUG('debug')
LOG_INFO('info')
LOG_WARN('warn')
LOG_ERROR('error')
LOG_CRITICAL('critical')

