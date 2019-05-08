
from PIL import Image

def make_img_cityscapes(output_data2,count,height,width) :
    catcolor2 = [(128, 64,128),(244, 35,232),(70,70,70),
    (102,102,156),(190,153,153),(153,153,153),(250,170,30),(220,220,0),
    (107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32),(0,0,0)]
    image = Image.new("RGB",(width,height))
    for x in range(width) :
        for y in range(height) :
            image.putpixel((x,y),catcolor2[output_data2[y][x]])
    filename = "outputs/" + str(count)+".png"
    image.save(filename)
