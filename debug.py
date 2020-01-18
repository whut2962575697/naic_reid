from common.optimizers import LRScheduler,WarmupMultiStepLR

if __name__ == '__main__':
    import torch
    from torchvision.models import resnet18
    print(torch.__version__)
    net = resnet18()
    optimizer = torch.optim.SGD(net.parameters(), 0.1)
    scheduler = WarmupMultiStepLR(optimizer, milestones=[7, 9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
    for i in range(10):
        print(i, scheduler.get_lr())
        print(i, optimizer.param_groups[0]['lr'])
        # optimizer.step()
        scheduler.step()