import cv2
import torch
from Mini_FAS import MultiFTNet
from MTCNN import MTCNN 
import torch.nn.functional as F
import numpy as np 
from albumentations import Compose, Normalize, Resize
import random


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg


def pre_process(image, image_size):
    pre_process_transform = Compose(
            [
                Resize(height=image_size[0], width=image_size[1]),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ]
        )
    return pre_process_transform(image=image)["image"].transpose((2, 0, 1))

class AntiFace:
    def __init__(self) -> None:
        self.model = MultiFTNet(num_classes=2)
        self.model.load_state_dict(torch.load('./checkpoint/FAS_best_score.pth'))
        self.model.eval()
        self.model.to('cuda')


    def Inference(self,Image):
        # Image = cv2.cvtColor(Image,cv2.COLOR_BGR2RGB)
        image_pre = pre_process(Image,image_size=(80,80))
        image_tensor = torch.from_numpy(image_pre)
        out = self.model.model.forward(image_tensor[None].to('cuda'))
        pred = F.softmax(out[0], dim=0)
        pred = pred.detach().cpu().numpy()
        # print(LABEL[np.argmax(pred)])
        return np.argmax(pred)

LABEL = ["FAKE","REAL"]
if __name__ == '__main__': 
    model = MultiFTNet(num_classes=2)
    model.load_state_dict(torch.load('./checkpoint/best_acc check_point_best_mode.pth'))
    model.eval()
    model.to('cuda')
    detect_model = MTCNN()
    webcam = cv2.VideoCapture(0)
    while True: 
        flag, frame = webcam.read()
        frame = cv2.flip(frame,1)
        if flag: 
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes, _ = detect_model.detect(image)
            if bboxes is not None: 
                for bbox in bboxes: 
                    x1, y1, x2, y2 = np.clip(bbox.astype(int), a_min=0, a_max=100000)
                    face_crop = image[y1:y2, x1:x2, :]
                    image_pre = pre_process(face_crop, image_size=(80,80))
                    image_tensor = torch.from_numpy(image_pre)
                    out = model(image_tensor[None].to('cuda'))
                    
                    pred = F.softmax(out[0], dim=0)
                    pred = pred.detach().cpu().numpy()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, LABEL[np.argmax(pred)], (x1, y1 - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # cv2.putText(frame, 'Real {:.4f}'.format(pred[1]), (x1, y1 - 50),
                    #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.imshow('Webcam', frame)
            cv2.imshow('',frame)
                    
            if cv2.waitKey(1) & 0xFF == 27: 
                break