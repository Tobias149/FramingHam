from os import path
import tkinter as tk
from tkinter.constants import ANCHOR, COMMAND, RIGHT, Y
from typing import BinaryIO, Text
import urllib
from numpy import common_type
from PIL import Image, ImageTk
from numpy.lib.polynomial import roots
import urllib.request
import numpy as np

## Funktion zur Schaltfläche Ende
def ende():
    fenster.destroy()

## Function submit button
def submit():

    second = tk.Tk()
    #back_second = tk.Canvas(second)
    #back_second.pack(expand=True, fill='both')
    #path1 = '/Users/tobiasschmidt/Desktop/healthy-heart.jpg'
    #back_second_image = Image.open(path1)
    #resize_back_second_image = back_second_image.resize((1100, 600))
    #back_second_image = ImageTk.PhotoImage(resize_back_second_image)
    #back_second.create_image(0,0, anchor='center', image = back_second_image)
    

    #VSex = int(li_SEX.get())
    VAge = int(txt_AGE.get())
    #VEducation = int(li_Education.get())
    #VCSmoker = int(li_CSmoker,get())
    VCigsPday = int(txt_CigsPday.get())
    #VBPMeds = int(li_BPMeds.get())
    #VPrevalentStroke = int(li_PrevalentStroke.get())
    #VPrevalentHYP = int(li_PrevalentHYP.get())
    #VDiabetes = int(li_Diabetes.get())
    VTotChol = int(txt_TotChol.get())
    VSysBP = int(txt_SysBP.get())
    VDiaBP = int(txt_DiaBP.get())
    VBMI = int(txt_BMI.get())
    VHeartRate = int(txt_HeartRate.get())
    VGlucose = int(txt_Glucose.get())

    xSubmit = np.array([VAge,VCigsPday,VTotChol,VSysBP,
                        VDiaBP,VBMI,VHeartRate,VGlucose])
    print(xSubmit)

    lb = tk.Label(second, text=xSubmit)
    lb.pack()

    second.mainloop()

## Main Window
fenster= tk.Tk()
fenster.title("TechLabs - Group 11")

back_gnd = tk.Canvas(fenster)
back_gnd.pack(expand=True, fill='both')

## Read and open the Image from Desktop
#path2 = '/Users/tobiasschmidt/Desktop/15-heart-symptoms-s2-heart-disease-warning-sign.jpg'
#path1 = '/Users/tobiasschmidt/Desktop/internal_structures_1_0.jpg'
#back_gnd_image = Image.open(path1)

## Read and open the Image from Github
urllib.request.urlretrieve("https://raw.githubusercontent.com/Tobias149/FramingHam/main/internal_structures_1_0.jpg", "internal_structures_1_0.jpg")
back_gnd_image = Image.open("internal_structures_1_0.jpg")

## Reszie the Image
resize_back_gnd_image = back_gnd_image.resize((1100, 600))
back_gnd_image = ImageTk.PhotoImage(resize_back_gnd_image)

#back_gnd_image = ImageTk.PhotoImage(Image.open(path2))
back_gnd.create_image(0,0, anchor='nw', image= back_gnd_image)

lbl1 = tk.Label(fenster, text="Source of image: National Heart, Lung, and Blood Institute; National Institutes of Health; U.S. Department of Health and Human Services.")
lbl1 ["font"] = "Courier 10"
lbl1 ["bg"] = "#FFFFFF"
lbl1.pack() # show label

lbl2 = tk.Label(fenster, text="https://www.nhlbi.nih.gov/news/2011/conquering-cardiovascular-disease;01.08.2021")
lbl2 ["font"] = "Courier 10"
lbl2 ["bg"] = "#FFFFFF"
lbl2.pack() # show label

lbl3 = tk.Label(None, text="Predicting a ten year risk of coronary heart disease")
lbl3 ["font"] = "Courier 20"
lbl3 ["bg"] = "#FFFFFF"
back_gnd.create_window(550,25, window=lbl3, anchor='center') # show label in background

## Naming of features left hand
lbl_Sex = tk.Label(fenster, text="Sex ", anchor="e")
back_gnd.create_window(190,100, window=lbl_Sex, anchor='center', width=230)
lbl_Education = tk.Label(fenster, text="Age in years ", anchor="e")
back_gnd.create_window(190,150, window=lbl_Education, anchor='center', width=230)
lbl_Age = tk.Label(fenster, text="Education ", anchor="e")
back_gnd.create_window(190,200, window=lbl_Age, anchor='center', width=230)
lbl_CSmoker = tk.Label(fenster, text="Current smoker ", anchor="e")
back_gnd.create_window(190,250, window=lbl_CSmoker, anchor='center', width=230)
lbl_CigsPday = tk.Label(fenster, text="Cigerrets per day ", anchor="e")
back_gnd.create_window(190,300, window=lbl_CigsPday, anchor='center', width=230) 
lbl_BPMeds = tk.Label(fenster, text="Use of Anti-hypertensive medication", anchor="e") # Use of Anti-hypertensive medication 
back_gnd.create_window(190,350, window=lbl_BPMeds, anchor='center', width=230) 
lbl_PrevalentStroke = tk.Label(fenster, text="Prevalent Stroke ", anchor="e") # 0 = Free of disease, 1 = Prevalent disease. 
back_gnd.create_window(190,400, window=lbl_PrevalentStroke, anchor='center', width=230)
lbl_PrevalentHYP = tk.Label(fenster, text="Prevalent Hypertensive ", anchor="e") # 0 = Free of disease, 1 = Prevalent disease.
back_gnd.create_window(190,450, window=lbl_PrevalentHYP, anchor='center', width=230)

## Naming of features right hand
lbl_Diabetes = tk.Label(fenster, text="Diabetes ", anchor="w") # 0 = Not a diabetic, 1 = Diabetic.'
back_gnd.create_window(900,100, window=lbl_Diabetes, anchor='center', width=230)
lbl_TotChol = tk.Label(fenster, text="Cholesterol [mg/dL] ", anchor="w")
back_gnd.create_window(900,150, window=lbl_TotChol, anchor='center', width=230)
lbl_SysBP = tk.Label(fenster, text="Systolic Blood Pressure [mmHg] ", anchor="w")
back_gnd.create_window(900,200, window=lbl_SysBP, anchor='center', width=230)
lbl_DiaBP = tk.Label(fenster, text="Diastolic Blood Pressure [mmHg] ", anchor="w")
back_gnd.create_window(900,250, window=lbl_DiaBP, anchor='center', width=230)
lbl_BMI = tk.Label(fenster, text="Body Mass Index ", anchor="w")
back_gnd.create_window(900,300, window=lbl_BMI, anchor='center', width=230)
lbl_HeartRate = tk.Label(fenster, text="Heart rate [beats/min] ", anchor="w")
back_gnd.create_window(900,350, window=lbl_HeartRate, anchor='center', width=230)
lbl_Glucose = tk.Label(fenster, text="Glucose level [mg/dL] ", anchor="w")
back_gnd.create_window(900,400, window=lbl_Glucose, anchor='center', width=230)

## Input features left hand
#frame_Sex =tk.Frame(fenster)
#scb_SEX = tk.Scrollbar(fenster,orient="vertical")
#li_SEX = tk.Listbox(fenster, height=0,yscrollcommand=scb_SEX.set)
#scb_SEX["command"]= li_SEX.yview
#Gender = ["female","male"] # 1 = Female, 0 = Male
#for i in Gender:
    #li_SEX.insert("end", i)
#back_gnd.create_window(335,100, window=li_SEX, anchor='center', width=60)

txt_AGE = tk.Entry(fenster)
txt_AGE.insert(0, "23") # years
back_gnd.create_window(335,150, window=txt_AGE, anchor='center', width=60) 

#scb_Education = tk.Scrollbar(fenster,orient="vertical")
#li_Education = tk.Listbox(fenster, height=1, yscrollcommand=scb_Education.set)
#scb_Education["command"]= li_Education.yview
#scb_Education.pack(side="left")
## 1 = 0-11 years, 2 = high school or GED, 3 = some college, 4 = college graduate or higher
#Education = ["primary school","high school","college","college grad."] 
#for i in Education:
    #li_Education.insert("end", i)
#back_gnd.create_window(355,200, window=li_Education, width=100)

#scb_CSmoker = tk.Scrollbar(fenster,orient="vertical")
#li_CSmoker = tk.Listbox(fenster, height=0,yscrollcommand=scb_CSmoker.set)
#scb_CSmoker["command"]= li_CSmoker.yview
#CSmoker = ["yes","no"]
#for i in CSmoker:
    #li_CSmoker.insert("end", i)
#back_gnd.create_window(335,250, window=li_CSmoker, anchor='center', width=60)

txt_CigsPday = tk.Entry(fenster)
txt_CigsPday.insert(0, "5")
back_gnd.create_window(335,300, window=txt_CigsPday, anchor='center', width=60) 

#scb_BPMeds = tk.Scrollbar(fenster,orient="vertical")
#li_BPMeds = tk.Listbox(fenster, height=0,yscrollcommand=scb_BPMeds.set)
#scb_BPMeds["command"]= li_BPMeds.yview
#BPMeds = ["yes","no"]
#for i in BPMeds:
    #li_BPMeds.insert("end", i)
#back_gnd.create_window(335,350, window=li_BPMeds, anchor='center', width=60)




## Input features right hand
txt_TotChol = tk.Entry(fenster)
txt_TotChol.insert(0, "190")
back_gnd.create_window(755,150, window=txt_TotChol, anchor='center', width=60) 
txt_SysBP = tk.Entry(fenster)
txt_SysBP.insert(0, "140") #mean systolic was >=140 mmHg
back_gnd.create_window(755,200, window=txt_SysBP, anchor='center', width=60) 
txt_DiaBP = tk.Entry(fenster)
txt_DiaBP.insert(0, "90") #mean Diastolic >=90 mmHg
back_gnd.create_window(755,250, window=txt_DiaBP, anchor='center', width=60)
txt_BMI = tk.Entry(fenster)
txt_BMI.insert(0, "23") # Body Mass Index, weight in kilograms/height meters squared.
back_gnd.create_window(755,300, window=txt_BMI, anchor='center', width=60) 
txt_HeartRate = tk.Entry(fenster)
txt_HeartRate.insert(0, "80")
back_gnd.create_window(755,350, window=txt_HeartRate, anchor='center', width=60)
txt_Glucose = tk.Entry(fenster)
txt_Glucose.insert(0, "70")
back_gnd.create_window(755,400, window=txt_Glucose, anchor='center', width=60) 

## Close button
cmd_button_e = tk.Button(None, text="Close", bd=1, highlightthickness=0, command= ende)
back_gnd.create_window(630,550, window=cmd_button_e, anchor='sw', width=150, height=50)

## Submit button
cmd_button_s = tk.Button(None, text="Submit", bd=1, highlightthickness=0, command= submit)
back_gnd.create_window(330,550, window=cmd_button_s, anchor='sw', width=150, height=50)

sLeft   =  "%s" % 500    # X-Position auf dem Bildschirm (linke obere Ecke in Pixels)
sTop    =  "%s" % 250    # Y-Position auf dem Bildschirm (linke obere Ecke in Pixels)
sWidth  =  "%s" % 1100   # Breite (Pixels)
sHeight =  "%s" % 600    # Höhe   (Pixels)

## Limitation of main window size
fenster.wm_geometry(sWidth+"x"+sHeight+"+"+sLeft+"+"+sTop)
fenster.resizable(width=0, height=0) # Verhinderung, dass die Fenstergröße verändert werden kann

## Loop end
fenster.mainloop()