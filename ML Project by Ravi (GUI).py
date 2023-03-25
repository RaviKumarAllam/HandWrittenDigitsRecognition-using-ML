#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tkinter as tk
import numpy as np
import cv2

#pip install pillow
from PIL import ImageTk,Image,ImageDraw

from keras.models import Sequential
from keras.layers import Dense,Flatten

model=Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.load_weights('https://github.com/RaviKumarAllam/HandWrittenRecognition-using-ML/blob/main/mnist.h5')
#C:/Users/Ravi Kumar/OneDrive/Desktop/6 weeks ML project-2/week 6/mnist.h5

def event_function(event):
    
    x=event.x
    y=event.y
    
    x1=x-30
    y1=y-30
    
    x2=x+30
    y2=y+30

    canvas.create_oval((x1,y1,x2,y2),fill='black')
    img_draw.ellipse((x1,y1,x2,y2),fill='white')

def save():
    
    global count
    
    img_array=np.array(img)
    img_array=cv2.resize(img_array,(28,28))
    
    cv2.imwrite(str(count)+'.jpg',img_array)
    count=count+1

def clear():
    
    global img,img_draw
    
    canvas.delete('all')
    img=Image.new('RGB',(500,500),(0,0,0))
    img_draw=ImageDraw.Draw(img)    
    
    label_status.config(text='PREDICTED DIGIT: NONE')

def predict():
    
    img_array=np.array(img)
    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array=cv2.resize(img_array,(28,28))
    
    img_array=img_array/255.0
    img_array=img_array.reshape(1,28,28)
    result=model.predict(img_array)
    label=np.argmax(result,axis=1)
    
    label_status.config(text='PREDICTED DIGIT:'+str(label))
    
count=0
    
win=tk.Tk()

canvas=tk.Canvas(win,width=500,height=500,bg='white')
canvas.grid(row=0,column=0,columnspan=4)

#button_save=tk.Button(win,text='SAVE',bg='green',fg='white',font='Helvetica 20 bold',command=save)
#button_save.grid(row=1,column=0)

button_predict=tk.Button(win,text='PREDICT',bg='gray',fg='white',font='Helvetica 20 bold',command=predict)
button_predict.grid(row=1,column=1)

button_clear=tk.Button(win,text='CLEAR',bg='green',fg='white',font='Helvetica 20 bold',command=clear)
button_clear.grid(row=1,column=2)

#button_exit=tk.Button(win,text='EXIT',bg='red',fg='white',font='Helvetica 20 bold',command=win.destroy)
#button_exit.grid(row=1,column=3)

label_status=tk.Label(win,text='PREDICTED DIGIT: NONE',bg='white',font='Helvetica 24 bold')
label_status.grid(row=2,column=0,columnspan=4)

canvas.bind('<B1-Motion>',event_function)
img=Image.new('RGB',(500,500),(0,0,0))
img_draw=ImageDraw.Draw(img)

win.mainloop()


# In[ ]:




