import argparse
import collections
import torch
import numpy as np
import loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

###############
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
###############

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # 데이터 로더 인스턴스 설정
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # 모델 아키텍처를 빌드한 다음 정보 출력
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # GPU 디바이스 설정(Multi-GPU 포함)
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # metric, loss 함수 불러오기
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # optimizer, learning rate scheduler를 빌드합니다.
    # ? 스케줄러를 비활성화하려면 lr_scheduler가 포함된 모든 줄 삭제.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Auto Trading')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config.json 파일 경로 (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='최근 체크포인트 경로 (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='활성화할 GPU의 인덱스 (default: all)')

    # 사용자 정의 cli 옵션을 사용하여 config.json 파일에 지정된 기본값을 수정할 수 있습니다.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    
    main(config)