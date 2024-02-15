import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

class BaseTrainer:
    """
    Base Trainer 클래스
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # 구성을 사용하여 모델 성능을 모니터링하고 최적의 모델을 저장합니다.
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # 시각화 인스턴스 설정                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Epoch 훈련 로직

        :param epoch: 현재 epoch
        """
        raise NotImplementedError

    def train(self):
        """
        전체 훈련 로직
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # 로그된 정보를 log 딕셔너리에 저장합니다.
            log = {'epoch': epoch}
            log.update(result)

            # 화면에 기록된 정보를 출력합니다.
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # 구성된 메트릭에 따라 모델 성능을 평가하고, 최적의 체크포인트를 model_best로 저장합니다.
            best = False
            if self.mnt_mode != 'off':
                try:
                    # 지정된 메트릭(mnt_metric)에 따라 모델 성능이 개선되었는지 여부를 확인합니다.
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}'을(를) 찾을 수 없습니다. "
                                        "모델 성능 모니터링이 비활성화됩니다.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("{} 에포크 동안 Validation 성능이 개선되지 않았습니다. "
                                     "훈련이 중지됩니다.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        체크포인트 저장

        :param epoch: 현재 Epoch
        :param log: Epoch 로깅 정보
        :param save_best: True일 경우, 저장된 체크포인트의 이름을 'model_best.pth'로 변경합니다.
        """
        arch = ['type'](self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        저장된 체크포인트에서 훈련 재개

        :param resume_path: 훈련을 재개할 체크포인트의 경로
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # 체크포인트의 매개변수를 로드합니다.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: config.json 파일에 지정된 아키텍처 구성이 체크포인트의 구성과 다릅니다. 이로 인해 state_dict가 로드되는 동안 예외가 발생할 수 있습니다.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # 현재 config['optimizer']['type']이 변경되지 않은 경우에만 체크포인트의 config['optimizer']['type']를 로드합니다.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: 현재 구성 파일에 저장된 config['optimizer']['type']이 체크포인트의 config['optimizer']['type']과 다릅니다."
                                "Optimizer parameter가 다시 시작되지 않았습니다.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("체크포인트가 로드되었습니다. 에포크 {}부터 훈련을 재개합니다".format(self.start_epoch))