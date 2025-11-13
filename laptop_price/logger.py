import logging
from pathlib import Path
LOG_FILE = Path('logs')
LOG_FILE.mkdir(exist_ok=True)


logging.basicConfig(
filename=LOG_FILE / 'project.log',
level=logging.INFO,
format='[%(asctime)s] %(levelname)s - %(message)s',
datefmt='%Y-%m-%d %H:%M:%S'
)


def get_logger(name=__name__):
    return logging.getLogger(name)