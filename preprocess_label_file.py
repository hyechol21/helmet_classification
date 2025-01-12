# Classification을 위한 라벨링 데이터 전처리
# [ 사람 / 어지러움 / 쓰러짐 / 폭력 / 위협 ]
# 1) 파일을 라벨링된 bbox crop
# 2) crop된 이미지를 클래스별로 폴더 나누기
# 3) 파일 이름 변경하여 저장 [index]_[class 이름].jpg

# 4) train, test 데이터 나누기 (각 클래스 폴더에서 랜덤으로 8:2)
# * train, test에 클래스별 비율이 동일

import os
import random

import cv2
from sklearn.model_selection import train_test_split


def get_file_list(folder_path):
    file_list = [file[:-4] for file in os.listdir(folder_path) if file.endswith(('.txt'))]
    file_list.sort()
    return file_list

# def split_labeling_file(file_list):
#     for file in file_list:
#         path_img = os.path.join(INPUT_PATH, file + '.jpg')
#         path_meta = os.path.join(INPUT_PATH, file + '.txt')
#
#         fr = open(os.path.join(path_meta), 'r')
#         img = cv2.imread(path_img)
#
#         height, width, _ = img.shape
#
#         lines = fr.readlines()
#         for line in lines:
#             box = list(map(float, line.split()))
#             box[0] = int(box[0])
#             box[1] = int((box[1] - box[3] / 2) * width)
#             box[2] = int((box[2] - box[4] / 2) * height)
#             box[3] = int(box[3] * width)
#             box[4] = int(box[4] * height)
#
#             # print(box)
#             idx, x, y, w, h = check_roi(width, height, box)
#
#             cropped_image = img[y: y+h, x: x+w].copy()
#
#             crop_filename = f'{new_file_count[idx]}_{classes[idx]}.jpg'
#             crop_path = os.path.join(folder_by_class[idx], crop_filename)
#             new_file_count[idx] += 1
#
#             try:
#                 cv2.imwrite(crop_path, cropped_image)
#             except:
#                 pass
#
#         fr.close()

def split_labeling_file(file_list):
    for file in file_list:
        path_img = os.path.join(INPUT_PATH, file + '.jpg')
        path_meta = os.path.join(INPUT_PATH, file + '.txt')

        fr = open(os.path.join(path_meta), 'r')
        img = cv2.imread(path_img)

        height, width, _ = img.shape

        person_list = []
        motion_list = []

        lines = fr.readlines()
        for line in lines:
            box = list(map(float, line.split()))
            box[0] = int(box[0])
            box[1] = int((box[1] - box[3] / 2) * width)
            box[2] = int((box[2] - box[4] / 2) * height)
            box[3] = int(box[3] * width)
            box[4] = int(box[4] * height)
            # print(box)
            roi = check_roi(width, height, box)
            if box[0] == 0:
                person_list.append(roi)
            else:
                motion_list.append(roi)
                crop_image(img.copy(), roi)

        for person in person_list:
            for motion in motion_list:
                iou = get_iou(person, motion)

                if iou > 0.9:
                    # print(person, iou)
                    break
            else:
                crop_image(img.copy(), person)

        fr.close()

def crop_image(img, roi):
    idx, x, y, w, h = roi
    cropped_image = img[y: y+h, x: x+w].copy()

    crop_filename = f'{new_file_count[idx]}_{classes[idx]}.jpg'
    crop_path = os.path.join(folder_by_class[idx], crop_filename)
    new_file_count[idx] += 1

    try:
        cv2.imwrite(crop_path, cropped_image)
    except:
        pass

def get_iou(box1, box2):
    box1_area = (box1[3] + 1) * (box1[4] + 1)
    box2_area = (box2[3] + 1) * (box2[4] + 1)

    # intersection
    x1 = max(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    x2 = min(box1[1]+box1[3], box2[1]+box2[3])
    y2 = min(box1[2]+box2[4], box2[2]+box2[4])

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def check_roi(img_width, img_height, box):
    idx, x,y,w,h = box
    x = x if (x <= img_width - w) else (img_width - w)
    y = y if (y <= img_height - h) else (img_height - h)
    x = int(x) if (x >= 0) else 0
    y = int(y) if (y >= 0) else 0
    box = (idx, x, y, w, h)
    return box

# train.txt 파일 생성
def create_image_path(path):
    train = []
    valid = []
    for folder_path in folder_by_class:
        num = 5000
        split_num = 4000

        file_list = os.listdir(folder_path)

        if folder_path == NORMAL_PATH:
            num = 10000
            split_num = 8000

        random_file_list = [os.path.join(folder_path, x) for x in random.sample(file_list, num)]

        train += random_file_list[:split_num]
        valid += random_file_list[split_num:]

    with open(os.path.join(path,'train.txt'), 'w') as f:
        for file in train:
            img_path = os.path.join(folder_path, file)
            f.write(img_path + '\n')

    with open(os.path.join(path, 'valid.txt'), 'w') as f:
        for file in valid:
            img_path = os.path.join(folder_path, file)
            f.write(img_path + '\n')

if __name__== '__main__':
    INPUT_PATH = './img'
    NORMAL_PATH = './normal/'
    DIZZY_PATH = './dizzy/'
    FALL_PATH = './fall/'
    FIGHT_PATH = './fight/'
    THREATEN_PATH = './threaten/'
    ETC_PATH = './etc/'

    TRAIN_PATH = './data/'

    classes = {0: 'normal', 1: 'dizzy', 2: 'fall', \
               3: 'fight', 4: 'threaten'}
    folder_by_class = [NORMAL_PATH, DIZZY_PATH, FALL_PATH, FIGHT_PATH, THREATEN_PATH]
    new_file_count = [0] * 5

    create_image_path(TRAIN_PATH)
