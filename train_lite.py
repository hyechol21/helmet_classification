# utils
import os
import numpy as np
import random
import copy
import time

# display
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter

# model
import torch
import torch.nn as nn
import torch.optim as optim

# from efficientnet_pytorch import EfficientNet
from efficientnet_lite_pytorch import EfficientNet
# from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile



## 데이터 로드
def load_dataset():
    ## make dataset
    from torchvision import transforms, datasets

    helmet_dataset = datasets.ImageFolder( data_path,
                                            transforms.Compose([
                                                transforms.Resize((image_size, image_size)),
                                                # transforms.ColorJitter(brightness=.5, hue=.3, saturation=.3),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ]))

    ## split data
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    train_idx, tmp_idx = train_test_split(list(range(len(helmet_dataset))), test_size=0.1, random_state=random_seed)

    datasets = {}
    datasets['train'] = Subset(helmet_dataset, train_idx)
    tmp_dataset = Subset(helmet_dataset, tmp_idx)

    val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed)
    datasets['valid'] = Subset(helmet_dataset, val_idx)
    datasets['test'] = Subset(helmet_dataset, test_idx)

    ## Data Loader 선언
    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                        batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                                        batch_size=batch_size, shuffle=False,
                                                        num_workers=num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'],
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers)

    batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(dataloaders['valid']), len(dataloaders['test'])
    print('batch_size : %d, tvt : %d / %d / %d' % (batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))
    return dataloaders


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)


## 데이터 체크
def check_dataset():
    import torchvision
    num_show_img = len(class_names)

    # check train
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # check valid
    inputs, classes = next(iter(dataloaders['valid']))
    out = torchvision.utils.make_grid(inputs[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # check test
    inputs, classes = next(iter(dataloaders['test']))
    out = torchvision.utils.make_grid(inputs[:num_show_img])
    imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])




## 결과 그래프 출력
def train_result_graph(best_idx, train_loss, train_acc, valid_loss, valid_acc):
    print('best model : %d - %1.f / %.1f' %(best_idx, valid_acc[best_idx], valid_loss[best_idx]))

    fig, ax1 = plt.subplots()
    ax1.plot(train_acc, 'b-')
    ax1.plot(valid_acc, 'r-')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-')
    ax2.plot(valid_loss, 'k-')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    plt.show()


## 학습
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_idx, best_acc = 0, 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train (학습시에만 히스토리를 추적)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) # batch의 평균 loss 출력

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu()*100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                writer_train.add_scalar('loss', epoch_loss, epoch)
                writer_train.add_scalar('accuracy', epoch_acc, epoch)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                writer_valid.add_scalar('loss', epoch_loss, epoch)
                writer_valid.add_scalar('accuracy', epoch_acc, epoch)

            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))
                save_name = '{}_{}_{:.1f}.pt'.format(model_name, epoch, best_acc)
                save_path = os.path.join(backup, save_name)
                torch.save(best_model_wts, save_path)

        # Early Stopping
        if epoch - best_idx >= 10:
            last_model_wts = copy.deepcopy(model.state_dict())
            print('==> Last model saved - %d / %.1f' % (epoch, epoch_acc))
            save_name = '{}_last_{:.1f}.pt'.format(model_name, best_acc)
            save_path = os.path.join(backup, save_name)
            torch.save(last_model_wts, save_path)
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


def test_model(weight, phase = 'test'):
    # phase = ' train', 'valid', 'test'
    model = EfficientNet.from_pretrained(model_name=model_name, weights_path=weight, num_classes=len(class_names))
    model.load_state_dict(torch.load(weight))
    model = model.to(device)
    model.eval()

    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size

        test_loss = running_loss / num_cnt
        test_acc = running_corrects.double() / num_cnt
        print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc * 100))



if __name__=='__main__':
    # class_names = {
    #     "0": "normal",
    #     "1": "fall",
    # }

    class_names = {
        "0": "helmet",
        "1": "no helmet",
        "2": "negative"
    }


    data_path = '/home/helena/hyewon/backup/dataset/0. helmet_final/train_0'

    # model save path
    backup = '/home/helena/PycharmProjects/EfficientNet_classification/backup/helmet_final_4'
    if not os.path.isdir(backup):
        os.makedirs(backup)

    ## set
    model_name = 'efficientnet-lite2'
    weights_path = EfficientnetLite2ModelFile.get_model_file_path()
    # weights_path = 'weight/efficientnet_lite3.pth'
    model = EfficientNet.from_pretrained(model_name, weights_path=weights_path, num_classes=len(class_names))
    # model = EfficientNet.from_pretrained(model_name, num_classes=len(class_names))
    # model.load_state_dict(torch.load(weights_path))
    batch_size = 32
    epochs = 100
    learning_rate = 0.001


    # Tensorboard 실시간 학습 보기
    # tensorboard --logdir=runs (해당 폴더위치에서 command)
    writer_train = SummaryWriter('runs/training')
    writer_valid = SummaryWriter('runs/validation')

    ## load model
    image_size = model.input_image_size
    print(f'input size: {image_size} x {image_size}')

    num_workers = 4 * torch.cuda.device_count()
    random_seed = 777
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set gpu

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()   # 다중 분류 모델에 사용

    # optimizer_ft = optim.SGD(model.parameters(),
    #                          lr=learning_rate,
    #                          momentum=0.9,
    #                          weight_decay=1e-4)

    optimizer_ft = optim.Adam(model.parameters(),
                              lr=learning_rate)

    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

    dataloaders = load_dataset()
    check_dataset()

    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model,
                                                                                          criterion,
                                                                                          optimizer_ft,
                                                                                          exp_lr_scheduler,
                                                                                          num_epochs=epochs,
                                                                                          )
    train_result_graph(best_idx, train_loss, train_acc, valid_loss, valid_acc)


    # weight = './backup/helmet_binary_3/efficientnet-lite2_20_99.8.pt'
    # test_model(weight, phase='test')

    writer_train.close()
    writer_valid.close()