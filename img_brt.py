from PIL import Image, ImageEnhance
import os

import cv2


def isImage(fil:str):
    fil = fil.lower()
    if any([fil.endswith(".jpg"),fil.endswith(".jpeg"),fil.endswith(".png")]):
        return True
    else:
        return False


def getNum(fil:str):
    num = (fil.split("."))[0][-3:]
    
    while not num[0].isdigit():
        num = num[1:]
    
    return int(num)

logs = open("logs.txt",'w')

count = 0

# for img in os.listdir("./photos/"):       
#     if isImage(img):
#         num = getNum(img)
#         logs.write(str(num))
#         logs.write('\n')

logs.close()

# file = "md34.jpg"
# img = Image.open(file)

# # print(img.save())

# random = 0

# # alt = ImageEnhance.Brightness(img)
# alt = ImageEnhance.Color(img)

# # alt = ImageEnhance.Contrast(img)

# new_image = alt.enhance(2.5)

# new_image.show()

# new_image.save(file)