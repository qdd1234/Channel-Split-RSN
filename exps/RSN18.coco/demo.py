#this file provide a demo which can use our model to predict single image
import os
import numpy as np
import cv2
import torchvision.transforms as transforms

import torch

from config import cfg
from network import RSN

from lib.utils.transforms import flip_back


def main():
    data_txt = np.loadtxt('./testImages.txt', dtype='str')    # image path
    # cfg
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    color_rgb = False
    border = 10
    kernel = 5
    shifts = [0.25]
    model_file = os.path.join('./RSN18.coco', "iter-{}.pth".format(0))  #model path please change to your own model path
    pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
             [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
             [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    color = np.random.randint(0, 256, (cfg.DATASET.KEYPOINT.NUM, 3)).tolist()

    model = RSN(cfg)
    if os.path.exists(model_file):
        state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    device = torch.device("cuda", 0)
    model.to(device)
    cpu_device =  torch.device("cpu")

    model.eval()

    for image_path in data_txt:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        ori_image = image.copy()

        height = ori_image.shape[0]
        width = ori_image.shape[1]

        if color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]))
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
            output = output.to(cpu_device).numpy()

            if cfg.TEST.FLIP:
                images_flipped = np.flip(image.to(cpu_device).numpy(), 3).copy()
                images_flipped = torch.from_numpy(images_flipped).to(device)
                output_flipped = model(images_flipped)
                output_flipped = output_flipped.to(cpu_device).numpy()
                output_flipped = flip_back(output_flipped, cfg.DATASET.KEYPOINT.FLIP_PAIRS)

        output = (output + output_flipped) * 0.5

        pred = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
        score = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))

        score_map = output[0].copy()
        score_map = score_map / 255 + 0.5

        dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
                           cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
        dr[:, border: -border, border: -border] = output[0].copy()
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
            # a = np.mat(dr[w])
            # cv2.imshow('a',a)
            # cv2.waitKey(0)
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))

            pred[w] = np.array([x * 4 + 2, y * 4 + 2])
            score[w, 0] = score_map[w, int(round(y) + 1e-9), \
                                    int(round(x) + 1e-9)]

        # aligned or not ...pred[:, 1] * h / cfg.INPUT_SHAPE[0]
        pred[:, 0] = pred[:, 0] * width / cfg.INPUT_SHAPE[1]
        pred[:, 1] = pred[:, 1] * height / cfg.INPUT_SHAPE[0]

        pred = pred.astype(int)

        joints = pred.copy()
        print(score.shape)
        score = score.squeeze().mean(axis=0)
        print(score.dtype)

        for i in range(cfg.DATASET.KEYPOINT.NUM):
            if joints[i, 0] > 0 and joints[i, 1] > 0:
                    cv2.circle(ori_image, tuple(joints[i, :2]), 2, tuple(color[i]), 2)
        if score:
            cv2.putText(ori_image, str(score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, \
                        (128, 255, 0), 2)

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(ori_image, joints[pair[0] - 1], joints[pair[1] - 1])



        cv2.imshow('pic', ori_image)
        cv2.waitKey(0)



if __name__=='__main__':
    main()
