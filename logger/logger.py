import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    로깅 구성 설정
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # 실행 구성에 따라 로깅 경로 수정하기
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logger_config 파일을 찾을 수 없습니다. {}.".format(log_config))
        logging.basicConfig(level=default_level)