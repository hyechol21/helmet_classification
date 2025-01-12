# 단일 크롭 이미지를 --> 분류 예측

import json
import os
import cv2
import time

import torch
# from torch2trt import torch2trt
from torchvision import transforms
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile


def load_model(device):
    # image_size = EfficientNet.get_image_size(model_name)

    model = EfficientNet.from_pretrained(model_name=model_name, weights_path=weight, num_classes=len(labels_map))
    # model = EfficientNet.from_pretrained(model_name, num_classes=len(labels_map))
    model.load_state_dict(torch.load(weight))
    model = model.to(device)
    model.eval()

    image_size = model.input_image_size

    # x = torch.ones((1, 3, image_size, image_size)).to(device)
    # model_trt = torch2trt(model, [x], use_onnx=True)
    # model_trt.eval()

    return model, image_size


def preprocess():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform


def predict_image_cv2(image_path):
    global tfms

    image = cv2.imread(image_path)
    # apply transforms to the input image
    input = tfms(image).unsqueeze(0)

    input = input.to(device)

    with torch.no_grad():
        outputs = model(input)

    preds = torch.topk(outputs, k=2).indices.squeeze(0).tolist()
    # print(preds)

    img = cv2.resize(image.copy(), (480, 480), interpolation=cv2.INTER_CUBIC)
    # img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)    # tensor to numpy

    print('-'*10)
    for i, ans_idx in enumerate(preds):
        # print(ans_idx)
        label = labels_map[ans_idx]
        prob = torch.softmax(outputs, dim=1)[0, ans_idx].item()
        print('{:<75} ({:.2f}%)'.format(label, prob*100))

        cv2.putText(img, '{}'.format(label), (15, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "{:.2f}%".format(prob*100), (200, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Inference result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # define the outfile file name
    filename = image_path.split('/')[-1]
    save_filename = f"outputs/result_{filename}"
    cv2.imwrite(save_filename, img)


def check_fps():
    global tfms

    image = cv2.imread(image_path)
    # apply transforms to the input image
    input = tfms(image).unsqueeze(0)

    input = input.to(device)

    start_time = time.time()
    with torch.no_grad():
        for i in range(1000):
            outputs = model(input)
    end_time = time.time()

    forward_time = (end_time - start_time) / 1000
    fps = int(1 / forward_time)

    print('Forward pass time(s): {:.3f}'.format(forward_time))
    print('fps: {}'.format(fps))


def predict_image_folder(folder_path):
    global tfms

    cnt = {'helmet': 0, 'no helmet': 0}
    with torch.no_grad():
        for idx, file in enumerate(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)
            print(image_path)
            # apply transforms to the input image
            input = tfms(image).unsqueeze(0)

            input = input.to(device)

            outputs = model(input)

            preds = torch.topk(outputs, k=1).indices.squeeze(0).tolist()

            img = cv2.resize(image.copy(), (480, 480), interpolation=cv2.INTER_CUBIC)
            # img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)    # tensor to numpy

            print('-'*10)
            for i, ans_idx in enumerate(preds):
                label = labels_map[ans_idx]

                if label == 'helmet':
                    cnt['helmet'] += 1
                elif label == 'no helmet':
                    cnt['no helmet'] += 1

                prob = torch.softmax(outputs, dim=1)[0, ans_idx].item()
                print('{:<75} ({:.2f}%)'.format(label, prob*100))

                cv2.putText(img, '{}'.format(label), (15, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, "{:.2f}%".format(prob*100), (200, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)


            # define the outfile file name
            filename = file.split('.')[0]
            save_filename = f"outputs/result_{idx}.png"
            cv2.imwrite(save_filename, img)
        print(cnt)



if __name__ == '__main__':

    image_path = '/home/helena/바탕화면/helmet_test'


    # load class names
    labels_map = ["helmet", "no helmet"]


    model_name = 'efficientnet-lite2'
    weight = './backup/helmet_binary_3/efficientnet-lite2_13_99.8.pt'
    # weight = './backup/helmet0/efficientnet-lite2_8_99.8.pt'


    # set the computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # load model
    model, image_size = load_model(device)


    # initialize the image transforms
    tfms = preprocess()


    predict_image_folder(image_path)
    # predict_image_cv2(image_path)
    # check_fps()