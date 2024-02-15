import importlib
from datetime import datetime

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # vizualization writer를 가져옵니다.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: 시각화(텐서보드)를 사용하도록 구성되었지만 현재 이 머신에 설치되어 있지 않습니다. " \
                    "'pip install tensorboardx'를 설치하거나, PyTorch --version >= 1.1로 업그레이드하여 'torch.utils.tensorboard'를 사용하세요." \
                    "시각화(텐서보드)를 사용하지 않으려면 'config.json' 파일에서 옵션을 해제해 주세요."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        시각화가 다음을 사용하도록 구성된 경우:
            추가 정보(step, tag)가 추가된 텐서보드의 add_data() 메서드를 반환합니다.
        그 :
            아무 작업도 수행하지 않는 함수를 반환합니다.
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # 이 클래스에 정의된 메서드, 예를 들어 set_step()을 반환하는 기본 로직입니다.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr