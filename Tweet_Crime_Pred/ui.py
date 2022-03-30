import PySimpleGUI as sg
import os
from predict import *

sg.theme("DarkBlue15")
layout = [
          [sg.Text('Specify Tweet', size=(15, 1), justification='right'), sg.InputText('SHE LIVED STREAMED WHILE TWO THUGS BOUND AND ATTACKED', key="tweet")],
          [sg.T("         "), sg.Radio('TF-IDF + KNN', "RADIO1", default=True, key="E")],
          [sg.T("         "), sg.Radio('TF-IDF + SVM', "RADIO1", default=False, key="P")],
          [sg.T("         "), sg.Radio('Jaccard + DNN', "RADIO1", default=False, key="X")],          
          [sg.T("        "), sg.Button('Testing >>>',size=(16,3), key="TE")]
         ]

###Setting Window
window = sg.Window('A Framework to Predict Social Crime through Twitter Tweets By Using Machine Learning', layout, size=(650,250))

###Showing the Application, also GUI functions can be placed here.

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        break    
    elif 'TE' in event:        
        sys = 'e'
        if values["E"] == True:
            sys ='KNN'
        if values["P"] == True:
            sys = 'SVM'
        if values["X"] == True:
            sys = 'DNN_3_Layer'
        tweet = values["tweet"]        
        predict(sys,tweet)
    
window.close()
