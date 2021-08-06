from os import path
import tkinter as tk
from tkinter.constants import ANCHOR, COMMAND
from typing import BinaryIO, Text
import urllib
from numpy import common_type
from PIL import Image, ImageTk
from numpy.lib.polynomial import roots

## Funktion zur Schaltfläche Ende
def ende():
    fenster.destroy()

## Main Window
fenster= tk.Tk()
fenster.title("TechLabs - Group 11")

back_gnd = tk.Canvas(fenster)
back_gnd.pack(expand=True, fill='both')

path1 = '/Users/tobiasschmidt/Desktop/internal_structures_1_0.jpg'
path2 = '/Users/tobiasschmidt/Desktop/15-heart-symptoms-s2-heart-disease-warning-sign.jpg'

## Read the Image of path1
back_gnd_image = Image.open(path1)

## Reszie the Image
resize_back_gnd_image = back_gnd_image.resize((900, 500))
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
back_gnd.create_window(450,25, window=lbl3, anchor='center') # show label in background

## Naming of features
lbl_Gender = tk.Label(fenster, text="Gender:", anchor="center")
back_gnd.create_window(55,100, window=lbl_Gender, anchor='center', width=150)
lbl_Glucose = tk.Label(fenster, text="Glucose level:", anchor="center")
back_gnd.create_window(55,150, window=lbl_Glucose, anchor='center', width=150)
lbl_Age = tk.Label(fenster, text="Age:", anchor="center")
back_gnd.create_window(55,200, window=lbl_Age, anchor='center', width=150)
lbl_BMI = tk.Label(fenster, text="Your BMI:", anchor="center")
back_gnd.create_window(55,250, window=lbl_BMI, anchor='center', width=150) 

## Input features
scb_Gender = tk.Scrollbar(fenster,orient="vertical")
li_Gender = tk.Listbox(fenster, height=0,yscrollcommand=scb_Gender.set)
scb_Gender["command"]= li_Gender.yview
Gender = ["female","male"]
for i in Gender:
    li_Gender.insert("end", i)
back_gnd.create_window(150,100, window=li_Gender, anchor='center', width=50)

txt_BMI = tk.Entry(fenster)
txt_BMI.insert(0, "23")
back_gnd.create_window(150,250, window=txt_BMI, anchor='center', width=50) 

txt_Glucose = tk.Entry(fenster)
txt_Glucose.insert(0, "")
back_gnd.create_window(150,150, window=txt_Glucose, anchor='center', width=50)

txt_Age = tk.Entry(fenster)
txt_Age.insert(0, "44")
back_gnd.create_window(150,200, window=txt_Age, anchor='center', width=50) 

## Close button
cmd_button = tk.Button(None, text="Close", bd=1, highlightthickness=0, command= ende)
back_gnd.create_window(800,430, window=cmd_button, anchor='sw')

sLeft   =  "%s" % 500    # X-Position auf dem Bildschirm (linke obere Ecke in Pixels)
sTop    =  "%s" % 250    # Y-Position auf dem Bildschirm (linke obere Ecke in Pixels)
sWidth  =  "%s" % 900    # Breite (Pixels)
sHeight =  "%s" % 500    # Höhe   (Pixels)

## Limitation of the front window size
fenster.wm_geometry(sWidth+"x"+sHeight+"+"+sLeft+"+"+sTop)
fenster.resizable(width=0, height=0) # Verhinderung, dass die Fenstergröße verändert werden kann

## Loop end
fenster.mainloop()