from MTCNN import MTCNN
import cv2
from face_regconition_model import iresnet, base_transform
import pandas as pd
import os


DIR_SAVE = ["./Datasets/train_crop/True/","./Datasets/train_crop/Fake/"]

def extract_frames(model,lst_dir:list[list]):
    index = 0
    for j in range(2):
        frame_temp = os.path.join(DIR_SAVE[j],'img_{:05d}.jpg')
        cnt = 0
        # print("Class {}")
        for i in lst_dir[j]:
            cap = cv2.VideoCapture("Datasets/" + i)
            while (cap.isOpened()):
                flag,frame = cap.read()
                if flag:
                    frame_path = frame_temp.format(cnt+1)
                    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    bbox, _ = model.detect(img)
                    if bbox is not None:
                        try:
                            x1, y1, x2, y2 = int(bbox[0, 0]), int(bbox[0, 1]), int(bbox[0, 2]), int(bbox[0, 3])    
                            crop = img[y1:y2, x1:x2]
                            crop = cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
                            cv2.imwrite(frame_path,crop)
                            cnt +=1
                            print(f"video{i} frame {cnt}")
                        except: 
                            pass
                else:break
            cap.release()
        


if __name__  == "__main__":
    face_det = MTCNN()
    lst = []
    lst_dir = []
    ### Read file csv
    file1 = pd.read_csv("./Label/train_True.csv")
    ##############
    temp = file1.columns[0].replace('.zip','').replace('/+CASIA-FASD/','')
    lst_dir.append(temp)
    col = file1[file1.columns[0]]
    for i in col:
        lst_dir.append(i.replace('.zip','').replace('/+CASIA-FASD/',''))
    lst.append(lst_dir)
    lst_dir = []
    file1 = pd.read_csv("./Label/train_Fake.csv")
    temp = file1.columns[0].replace('.zip','').replace('/+CASIA-FASD/','')
    lst_dir.append(temp)
    col = file1[file1.columns[0]]
    for i in col:
        lst_dir.append(i.replace('.zip','').replace('/+CASIA-FASD/',''))
    lst.append(lst_dir)
    extract_frames(face_det,lst)
    
    
    
    
    