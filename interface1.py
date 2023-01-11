# -*- coding: utf-8 -*-
"""
Created on Fri apr 21 10:01:00 2022

@author: badr
"""
from gtts import gTTS
import sys
import Blocks 
from playsound import playsound
import speech_recognition as sr

import os
from csv import DictWriter
import csv
import keyboard
#import time
#import json

r = sr.Recognizer()
mic = sr.Microphone()
Y=Blocks.S_G()

def speak():
    a = keyboard.read_key()
    if a == '0':
        with mic as source:
            r.adjust_for_ambient_noise(source)
            Audio = r.listen(source)
            x= r.recognize_google(Audio,language="ar-EG")
        return x    

def new_person(name):
    if os.path.isfile("Data.csv"):
        with open('Data.csv', 'r') as f:
            reader = csv.reader(f)
            ID=len(list(reader))

    with open('Data.csv', 'a+',encoding='utf-8-sig', newline = '') as f:
        dict_writer = DictWriter(f, fieldnames=['Name','ID'])
        if os.stat('Data.csv').st_size == 0:        #if file is not empty than header write else not
            dict_writer.writeheader()
            ID=1
                
        dict_writer.writerow({'Name' : name,'ID': str(ID)})
        
    #with open('output.json','w',encoding='utf8') as f:
    #    json.dump(name,f,ensure_ascii=False)

    if not os.path.isdir("names"):
        os.mkdir("names")

    test = gTTS(" ،هذا الشخص هو"+name,lang='ar')
    test.save("names/"+str(ID)+".mp3")

    Y.register(ID)
    Y.training()
    return

def old_person(ID):
    with open('Data.csv', 'r',encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        #header = next(reader)
        #if header != None:
        i=0
        for row in reader:
            i = i + 1
            if int(row['ID']) == ID:
                guest=row
                playsound("names/"+str(ID)+".mp3")
                #print("this is : {}".format(guest["Name"]))
                return
            else: 
                continue
    
    playsound("commands/f_r3.mp3")
    return
    
def sys():

  name = ""
  ID = 0   
  comm = "commands/intro.mp3"

  while True:
     """"Say 1 for live object detection,
        Say 2 for face recognition,
        Say 3 for images text to speech conversion,
        say 0 for exitfrom system
        press esc. for exiting from the mode you enter
     """
     option = ""
     #playsound(comm)
     comm = "commands/repeat.mp3"
     #option = input("..Select mode.....")
     option = speak()

     if option == "اغراض": 
         
         Y.yolo_detect()
         #yolowebcam.test()
        
     elif option == "اشخاص":

        playsound("commands/f_r.mp3")
        s = speak()

        if s == "جديد":
            playsound("commands/f_r1.mp3")
            name = speak()
            new_person(name)
                
                
               
            
        elif s == "قديم":
            playsound("commands/f_r2.mp3")
            ID=Y.detect()
            old_person(ID)
                
            
        else:
            continue


                

         #Blocks.imagetospeech.test()
     elif option == "نصوص":
         Y.imagetospeech()
         
              
     elif option == "0":
         playsound("commands/end.mp3")
         return
         
     else:
         print('Wrong mode, Select Again please..')
         continue
 
if __name__ == '__main__':
    sys()