import os
import torch
import argparse

def parse_args():
    # ArgumentParser 객체 생성
    p = argparse.ArgumentParser(description="Example deep learning script")

    # 학습 파라미터 인자 추가
    p.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    p.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    # 인자를 파싱하고 반환
    args = p.parse_args()
    return args

def main():
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # Load data and split into train/valid dataset.
    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    # Get input/output size to build model for any dataset.
    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    # Build model using given configuration.
    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    # Initialize trainer object.
    trainer = Trainer(model, optimizer, crit)

    # Start train with given dataset and configuration.
    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)

    # 모델 학습 로직 등...

if __name__ == "__main__":
    config = parse_args()
    main(config)