#Run the Code when the game is opened and the player is in the vechicle

import mss
import time
import keyboard
import os

screenshotFolder = "screenshots"
incrementNumberFile="index.txt"
os.makedirs(screenshotFolder, exist_ok=True)
def screenCapture(index):
    with mss.mss() as sct:
        monitors=sct.monitors
        monitor=monitors[1]
        screenshot=sct.grab(monitor)
        mss.tools.to_png(screenshot.rgb,screenshot.size,output="screenshots/screenshot"+str(index)+".png")
if not os.path.exists(incrementNumberFile) or os.stat(incrementNumberFile).st_size ==0:
    fileWrite=open(incrementNumberFile,"w")
    fileWrite.write("1")
    fileWrite.close()
fileRead=open(incrementNumberFile,"r")
indexNumber=fileRead.read().strip()
fileRead.close()
i=1
if indexNumber.isdigit():
    i=int(indexNumber)
else:
    i=1
while True:
    if keyboard.is_pressed("w") or keyboard.is_pressed("a") or keyboard.is_pressed("s") or keyboard.is_pressed("d"):
        screenCapture(i)
        i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("esc"):
        fileWrite=open(incrementNumberFile,"w")
        fileWrite.write(str(i))
        fileWrite.close()
        break