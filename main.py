from tkinter import *
import cv2
from PIL import ImageTk, Image
import os

image_name = "image.png"


def create_picture():
    camera = cv2.VideoCapture(0)
    result, image = camera.read()
    if result:
        cv2.imwrite(image_name, image)


root = Tk()

root.geometry("600x600")

btn = Button(root, text="Create image", bd="5", command=create_picture)
btn.pack(side="top")

frame = Frame(root, width=600, height=400)
frame.pack()
frame.place(anchor="center", relx=0.5, rely=0.5)

img = ImageTk.PhotoImage(Image.open(image_name))

label = Label(frame, image=img)
label.pack()


root.mainloop()
