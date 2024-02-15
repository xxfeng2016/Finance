import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        클래스를 사용하여 구성 config.json 파일을 파싱합니다.
        트레이닝, 모듈 초기화, 체크포인트 저장 및 로깅 모듈을 위한 하이퍼파라미터를 처리합니다.
        :param config: 설정, 훈련용 하이퍼파라미터를 포함하는 json 객체입니다. 
        :param resume: String, 체크포인트의 경로를 지정합니다.
        :param modification: Dict keychain:value, config.json 에서 수정할 값을 지정합니다.
        :param run_id: 훈련 프로세스 id. 체크포인트와 training log를 저장하는 데 사용됩니다. 기본값은 타임스탬프입니다.
        """
        # 구성 파일 로드 및 수정 사항 적용
        self._config = _update_config(config, modification)
        self.resume = resume

        # 학습된 모델과 로그가 저장될 저장 디렉토리를 설정합니다.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # 타임스탬프를 기본 실행 ID로 사용
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # 체크포인트와 로그를 저장할 디렉터리를 만듭니다.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # 업데이트된 구성 파일을 체크포인트 디렉터리에 저장합니다.
        write_json(self.config, self.save_dir / 'config.json')

        # 로깅 모듈 구성
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        일부 CLI 인자로부터 이 클래스를 초기화합니다.  train, test에 사용됩니다.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "구성 파일을 지정해야 합니다. 예를 들어 '-c config.json'을 추가하세요."
            ##############################
            # assert args.config is not None, msg_no_cfg
            args.config = "config.json"
            ##############################
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        구성에서 'type'으로 지정된 이름을 가진 함수 핸들을 찾고, 주어진 인수를 사용하여 초기화된 인스턴스를 반환합니다.

        `object = config.init_obj('name', module, a, b=1)`은 다음과 같습니다.
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), '구성 파일에 제공된 kwarg 덮어쓰기가 허용되지 않습니다'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        구성에서 'type'으로 지정된 이름의 함수 핸들을 찾고, 주어진 인수를 가진 함수를 functools.partial로 고정하여 반환합니다.

        `function = config.init_ftn('name', module, a, b=1)`은 다음과 같습니다.
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), '구성 파일에 제공된 kwarg 덮어쓰기가 허용되지 않습니다'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {}이 유효하지 않습니다. 유효한 옵션은 {}입니다.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # 읽기 전용 속성 설정
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# 사용자 지정 CLI 옵션으로 구성 딕셔너리를 업데이트하는 helper functions
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """트리에서 중첩된 개체의 값을 키 시퀀스로 설정합니다."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """트리에서 중첩된 개체의 값을 키 시퀀스로 설정합니다."""
    return reduce(getitem, keys, tree)