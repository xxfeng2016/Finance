from torch.nn import CrossEntropyLoss

def cross_entropy(output, target):
    loss = CrossEntropyLoss()
    return loss(output, target)