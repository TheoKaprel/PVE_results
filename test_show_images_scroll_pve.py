from tkinter import *
from PIL import ImageTk, Image
import numpy as np

def nex_img(i):   # takes the current scale position as an argument
    # delete previous image
    canvas.delete('image')
    # create next image
    canvas.create_image(0, 0, anchor=NW, image=listimg[int(i)-1], tags='image')


size=(1920,1200)
size=(1000,600)
root = Tk()

# image1 = ImageTk.PhotoImage(Image.open('/export/home/tkaprelian/Desktop/PVE/Results/ex_proj_1.png').resize(size))
# image2 = ImageTk.PhotoImage(Image.open('/export/home/tkaprelian/Desktop/PVE/Results/ex_proj_2.png').resize(size))
# image3 = ImageTk.PhotoImage(Image.open('/export/home/tkaprelian/Desktop/PVE/Results/ex_proj_3.png').resize(size))


image1=ImageTk.PhotoImage(image=Image.fromarray(np.random.randint(1,255,size=(size[1], size[0],3),dtype=np.uint8)))
image2=ImageTk.PhotoImage(image=Image.fromarray(np.random.randint(1,255,size=(size[1], size[0],3),dtype=np.uint8)))
image3=ImageTk.PhotoImage(image=Image.fromarray(np.random.randint(1,255,size=(size[1], size[0],3),dtype=np.uint8)))

listimg = [image1, image2, image3]

scale = Scale(master=root, orient=HORIZONTAL, from_=1, to=len(listimg), resolution=1,
              showvalue=False, command=nex_img)
scale.pack(side=BOTTOM, fill=X)
canvas = Canvas(root, width=size[0], height=size[1])
canvas.pack()

# show first image
nex_img(1)

root.mainloop()