import argparse
import pathlib
from torch import nn, optim
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image

from create_backdoor_data_loader import create_backdoor_data_loader
from deeplearn import test, train
from models.Nets import alexnet
from models.resnet import resnet18
from models.vgg import vgg11

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='Which dataset to use (mnist or cifar10, default: mnist)')
parser.add_argument('--trigger_label', type=int, default=2, help='The NO. of trigger label (int, range from 0 to 10')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate of the model, default: 0.001')
parser.add_argument('--pp', action='store_true', help='Do you want to print performance of every label in every epoch (default false, if you add this param, then print)')
parser.add_argument('--datapath', default='./dataset/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--poisoned_portion', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--lr_decay_epochs', type=str, default='50,80,90', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

opt = parser.parse_args()
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# read dataset: %s " % opt.dataset)
    if opt.dataset == 'mnist':
        train_data = datasets.MNIST(root=opt.datapath, train=True, download=True)
        test_data = datasets.MNIST(root=opt.datapath, train=False, download=True)
    elif opt.dataset == 'cifar10':
        train_data = datasets.CIFAR10(root=opt.datapath, train=True, download=True)
        test_data = datasets.CIFAR10(root=opt.datapath, train=False, download=True)
    else:
        print("Have no this dataset")

    # for i in range(0, 10000):
    #     img, tag = test_data[i]
    #     tag = str(tag)
    #
    #     img.save("C:/Users/DN2/Desktop/img_mnist/" + tag +"/s{}.png".format(i))
    print("\n# construct poisoned dataset")
    train_data_loader, test_data_ori_loader, test_data_tri_loader = create_backdoor_data_loader(opt.dataset, train_data, test_data, opt.trigger_label, opt.poisoned_portion, opt.batchsize, device)

    print("\n# begin training backdoor model")
    # basic_model_path = "D:/研一下学期/pytorch-cifar10-main/weights/%s-alexnet_weights_100.pth" % opt.dataset
    model = resnet18(in_size=28, num_classes=10, grayscale=True)
    model = model.to(device)
    state_dict = torch.load('D:/研一下学期/pytorch-cifar10-main/weights/mnist-resnet18_weights_010.pth')
    model.load_state_dict(state_dict)

    # 冻结不需要训练的网络层
    i = 0
    for param in model.parameters():
        i += 1
        if i < 60:
            param.requires_grad = False


    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 80], gamma=0.1)
    for epoch in range(100):
        train(model, device, train_data_loader, criterion, optimizer, epoch)
        test(model, device, test_data_ori_loader, criterion)
        print("*"*20)
        test(model, device, test_data_tri_loader, criterion)
        torch.save(model.state_dict(), './logs/mnist/model_trj_params_{}.pth'.format(epoch))
        lr_sched.step()

    # if opt.no_train:
    #     model = backdoor_model_trainer(
    #             dataname=opt.dataset,
    #             train_data_loader=train_data_loader,
    #             test_data_ori_loader=test_data_ori_loader,
    #             test_data_tri_loader=test_data_tri_loader,
    #             trigger_label=opt.trigger_label,
    #             epoch=opt.epoch,
    #             batch_size=opt.batchsize,
    #             loss_mode=opt.loss,
    #             optimization=opt.optim,
    #             lr=opt.learning_rate,
    #             print_perform_every_epoch=opt.pp,
    #             basic_model_path=basic_model_path,
    #             device=device
    #             )
    #
    # print("\n# evaluation")
    # print("## original test data performance:")
    # print_model_perform(model, test_data_ori_loader)
    # print("## triggered test data performance:")
    # print_model_perform(model, test_data_tri_loader)
    # tag = 1
    # while(tag == 1):
    #     array2img(test_data_tri_loader)
    #     tag += 1

if __name__ == "__main__":
    main()

# img, tar = train_data[0]
#     trig_path = './trigger/s180.png'
#     trig = cv2.imread(trig_path)
#     # trig = trig.transpose((2, 0, 1))
#     w2, h2, _ = trig.shape
#     print(type(img))
#     img = np.array(img)
#
#     width, height, _ = img.shape
#     for i in range(width):
#         for j in range(height):
#             if (i >= 25 and i < (25 + w2)) and (j >= 25 and j < (25 + h2)):
#                 img[i, j, :] = trig[i - 25, j - 25, :]
#     img = Image.fromarray(img)
#     img.save("s.png")
#     print(type(tar))
