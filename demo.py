import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b2
from tqdm import tqdm
import numpy as np
import mss
import cv2
import keyboard
import time
from PIL import Image

actions = {
    0: ["w"],
    1: ["w", "a"],
    2: ["w", "d"],
    3: ["a"],
    4: ["d"],
    5: ["s"],
    6: ["space"]
}


def getimage():
    with mss.mss() as sct:
        monitors=sct.monitors
        monitor=monitors[2]
        screenshot=sct.grab(monitor)
        screenshot = Image.fromarray(np.array(screenshot)).convert("RGB")
        screenshot = screenshot.resize((480,270))
        return screenshot

def presskey(ids):
    for id in ['w', 'a', 's', 'd', 'space']:
        keyboard.release(id)
    for id in ids:
        keyboard.press(id)

def detect_stuck(prev_frame, curr_frame, threshold=0.05):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude = np.linalg.norm(flow, axis=2)
    avg_magnitude = np.mean(magnitude)
    
    if avg_magnitude < threshold:
        return True
    return False

def main():
    num_classes = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnet_b2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.eval()
    model.load_state_dict(torch.load("model.pt", mmap='cpu', weights_only=True))
    model.to(device)
    started = False
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    prev_frame = None
    cur_frame = None
    i = 0
    while True:
        if keyboard.is_pressed("q"):
            started = True
        if keyboard.is_pressed("esc"):
            break
        if not started: continue
        img = getimage()
        if i%2 == 0:
            prev_frame = np.array(img)
        if i%2 == 1:
            cur_frame = np.array(img)
        if i%3 == 2 and detect_stuck(prev_frame, cur_frame):
            choose_action = np.random.choice(['w_a', 'w_d', 's_a', 's_d'])
            a1, a2 = choose_action.split("_")
            presskey([a1,a2])
            time.sleep(1)
        else:
            tr_img = transform(img).to(device)
            outputs = model(tr_img[None])
            _, predicted = outputs.max(1)

            presskey(actions[predicted.item()])
            time.sleep(0.2)
        i+=1
if __name__ == '__main__':
    main()
