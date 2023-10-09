from tkinter import *
import cv2
from PIL import ImageTk, Image
import os

image_name = "camera.png" if os.path.exists("camera.png") else "assets/placeholder.jpg"


def create_picture():
    camera = cv2.VideoCapture(0)
    result, image = camera.read()
    if result:
        cv2.imwrite("camera.png", image)
        change_img()


def change_img():
    img2 = ImageTk.PhotoImage(Image.open("camera.png"))
    label.configure(image=img2)
    label.image = img2


root = Tk()

root.geometry("600x600")

btn = Button(root, text="Create image", bd="5", command=create_picture)
btn.pack(side="top")

frame = Frame(root, width=300, height=300)
frame.pack()
frame.place(anchor="center", relx=0.5, rely=0.5)

img = ImageTk.PhotoImage(Image.open(image_name))

label = Label(frame, image=img)
label.pack()

root.bind("<Return>", change_img)

root.mainloop()
