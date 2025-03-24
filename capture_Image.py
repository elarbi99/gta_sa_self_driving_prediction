#Run the Code when the game is opened and the player is in the vechicle

import mss
import time
import keyboard
import os

trainingFolder = "training_data"
incrementNumberFile="index.txt"
wLabelsFile="wLabels.txt"
aLabelsFile="aLabels.txt"
sLabelsFile="sLabels.txt"
dLabelsFile="dLabels.txt"
waLabelsFile="waLabels.txt"
wdLabelsFile="wdLabels.txt"
os.makedirs(trainingFolder, exist_ok=True)
def screenCapture(index,fileName,keyPressed):
    with mss.mss() as sct:
        monitors=sct.monitors
        monitor=monitors[1]
        screenshot=sct.grab(monitor)
        mss.tools.to_png(screenshot.rgb,screenshot.size,output="training_data/screenshot"+str(index)+".png")
    fileWrite=open(fileName,"a")
    fileWrite.write("screenshot"+str(i)+".png" + " Label: "+keyPressed+"\n")
    fileWrite.close()
    

def fileCheck(fileName):
    if not os.path.exists(fileName) or os.stat(fileName).st_size ==0:
        fileWrite=open(fileName,"w")
        if fileName == "index.txt":   
            fileWrite.write("1")
        fileWrite.close()
fileCheck(incrementNumberFile)
fileCheck(wLabelsFile)
fileCheck(aLabelsFile)
fileCheck(sLabelsFile)
fileCheck(dLabelsFile)
fileCheck(waLabelsFile)
fileCheck(wdLabelsFile)
fileRead=open(incrementNumberFile,"r")
indexNumber=fileRead.read().strip()
fileRead.close()
i=1
if indexNumber.isdigit():
    i=int(indexNumber)
else:
    i=1
while True:
    if keyboard.is_pressed("w") and keyboard.is_pressed("a"):
        screenCapture(i,waLabelsFile,"wa")
        i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("w") and keyboard.is_pressed("d"):
        screenCapture(i,wdLabelsFile,"wd")
        i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("w"):
        screenCapture(i,wLabelsFile,"w")
        i+=1
        time.sleep(0.2)
   
    elif keyboard.is_pressed("a"):
        screenCapture(i,aLabelsFile,"a")
        i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("s"):
        screenCapture(i,sLabelsFile,"s")
        i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("d"):
        screenCapture(i,dLabelsFile,"d")
        i+=1
        time.sleep(0.2)
    elif keyboard.is_pressed("esc"):
        fileWrite=open(incrementNumberFile,"w")
        fileWrite.write(str(i))
        fileWrite.close()
        break