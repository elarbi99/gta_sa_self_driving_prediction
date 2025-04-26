#Run the Code when the game is opened and the player is in the vechicle

import mss
import time
import keyboard
import os
from os.path import join
import cv2
import numpy as np

labels = {
    "w": "0_w",
    "wa": "1_wa",
    "wd": "2_wd",
    "a": "3_a",
    "d": "4_d",
    "s": "5_s",
    "space": "6_space"
}

folder = "test"
for v in labels.values():
    os.makedirs(join(folder, v), exist_ok=True)


def find_min(counts):
    min_index = np.argmin(counts)
    min_val = np.min(counts)
    return min_index, min_val


incrementNumberFile="index_train.txt" if folder == 'train' else 'index_test.txt'
def screenCapture(index,img_folder):
    with mss.mss() as sct:
        monitors=sct.monitors
        monitor=monitors[1]
        screenshot=sct.grab(monitor)
        screenshot = np.array(screenshot)
        screenshot = cv2.resize(screenshot, (480,270))
        cv2.imwrite(f"{img_folder}/screenshot"+str(index)+".png", screenshot)
    

def fileCheck(fileName):
    if not os.path.exists(fileName) or os.stat(fileName).st_size ==0:
        fileWrite=open(fileName,"w")
        fileWrite.write("1")
        fileWrite.close()
fileCheck(incrementNumberFile)
fileRead=open(incrementNumberFile,"r")
indexNumber=fileRead.read().strip()
fileRead.close()
i=1
if indexNumber.isdigit():
    i=int(indexNumber)
else:
    i=1

started = False
start = time.time()
while True:
    if keyboard.is_pressed("q"):
        started = True
    if not started:
        continue
    counts = [len(os.listdir(join(folder, v))) for v in labels.values()]
    min_index, min_val = find_min(counts)
    if int(time.time() - start)%5==0:
        print(counts)
    if keyboard.is_pressed("w") and keyboard.is_pressed("a"):
        if counts[int(labels['wa'].split("_")[0])] < 200:
            screenCapture(i,join(folder,labels['wa']))
            i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("w") and keyboard.is_pressed("d"):
        if counts[int(labels['wd'].split("_")[0])] < 200:
            screenCapture(i,join(folder,labels['wd']))
            i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("w"):
        if counts[int(labels['w'].split("_")[0])] < 200:
            screenCapture(i,join(folder,labels['w']))
            i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("a"):
        if counts[int(labels['a'].split("_")[0])] < 200:
            screenCapture(i,join(folder,labels['a']))
            i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("s"):
        if counts[int(labels['s'].split("_")[0])] < 200: 
            screenCapture(i,join(folder,labels['s']))
            i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("d"):
        if counts[int(labels['d'].split("_")[0])] < 200:
            screenCapture(i,join(folder,labels['d']))
            i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("space"):
        if counts[int(labels['space'].split("_")[0])] < 200:
            screenCapture(i,join(folder,labels['space']))
            i += 1
        time.sleep(0.2)
    elif keyboard.is_pressed("esc") or min_val == 200:
        fileWrite=open(incrementNumberFile,"w")
        fileWrite.write(str(i))
        fileWrite.close()
        break