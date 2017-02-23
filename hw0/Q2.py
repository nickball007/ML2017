from PIL import Image, ImageDraw
import math
import sys

from PIL import ImageChops

im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])

aa = im1.load()
bb = im2.load()
width, height =  im2.size
cc = Image.new("RGBA",(width, height),(0,0,0,0))

for j in range(height):
    for i in range(width):
        if im1.getpixel((i,j)) != im2.getpixel((i,j)):
            cc.putpixel((i,j), im2.getpixel((i,j)))



cc.save("aa.png")
