import os, sys, glob
from PIL import Image
import random

def resize_img(imgFile,newSize):
    '''
    imgFile: String. Is the absolute path to an image, or an JpegImageFile
    newSize: Integer Tuple. Is max pixel W and H of new image
    '''
    if isinstance(imgFile, str):
        img = Image.open(imgFile)
    else:
        img = imgFile
    img.thumbnail(newSize, Image.ANTIALIAS)
    return img


def paste_img_on_background(imgFile, backgroundSize):
    '''
    imgFile: String. Is the absolute path to an image, or an JpegImageFile
    backgroundSize: Integer Tuple. Is the size of the background onto which
    the image from imgFile is pasted
    '''
    if isinstance(imgFile, str):
        img = Image.open(imgFile)
    else:
        img = imgFile
    img_w, img_h = img.size
    background = Image.new('RGB', backgroundSize)
    bg_w, bg_h = background.size
    offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))
    background.paste(img, offset)
    return background


def convert_color_of_img(imgFile, color):
    '''
    imgFile: String. The absolute path to an image, or an JpegImageFile
    color: String. Is either 'RGB' or 'gray' or 'bw'
    '''
    if isinstance(imgFile, str):
        img = Image.open(imgFile)
    else:
        img = imgFile
    if color == 'gray':
        img = img.convert('L')
    elif color == 'RGB':
        img = img.convert('RGB')
    elif color == 'bw':
        img = img.convert('1')
    return img
    
def flip_image(imgFile, direction):
    '''
    imgFile: String. The absolute path to an image, or a JpegImageFile
    direction: String. Can be lr, tb, r90, r180, or r270, depending on the
        desired tranformations
    '''
    if isinstance(imgFile, str):
        img = Image.open(imgFile)
    else:
        img = imgFile
    if direction == 'lr':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'tb':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif direction == 'r90':
        img = img.transpose(Image.ROTATE_90)
    elif direction == 'r180':
        img = img.transpose(Image.ROTATE_180)
    elif direction == 'r270':
        img = img.transpose(Image.ROTATE_270)
    return img

def blend_two_images(imgFile1, imgFile2):
    '''
    imgFile(1|2): String or ImageFile 
                  The absolute path to an image, or an JpegImageFile
    '''
    if isinstance(imgFile1, str):
        background = Image.open(imgFile1)
        overlay = Image.open(imgFile2)
    else:
        background = imgFile1
        overlay = imgFile2

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    img = Image.blend(background, overlay, 0.5)
    return img

def do_all_the_transforms(imgFile, base, abspath):
    '''
    Input: 
      imgFile: a path to an image or an image object
      base: the basename of the filepath
      abspath: the absolute path to the file
    Output: 
      transformed and translated images are saved to same dir as org image
    '''
    # do relevant tranformations on image before translations
    smallImg = resize_img(imgFile, newSize=(128,128))
    squareImg = paste_img_on_background(smallImg,backgroundSize=(128,128))
    newImg = convert_color_of_img(squareImg,'gray')

    newImg = flip_image(img, 'lr')
    newImg.save(os.path.join(abspath, '_lr_' + base + '.jpg'), 'JPEG')

    newImg = flip_image(img, 'tb')
    newImg.save(os.path.join(abspath, '_tb_' + base + '.jpg'), 'JPEG')

    newImg = flip_image(img, 'r90')
    newImg.save(os.path.join(abspath, '_r90_' + base + '.jpg'), 'JPEG')

    newImg = flip_image(img, 'r180')
    newImg.save(os.path.join(abspath, '_r180_' + base + '.jpg'), 'JPEG')

    newImg = flip_image(img, 'r270')
    newImg.save(os.path.join(abspath, '_r270_' + base + '.jpg'), 'JPEG')

if __name__ == '__main__':
    dir = sys.argv[1]

    i=0
    allFiles = glob.glob(dir + '*.jpg')

    for infile1 in allFiles:
        newImg = convert_color_of_img(infile1, 'gray')
        newImg.save(infile1, 'JPEG')
        # sys.exit()
        # for infile2 in random.sample(allFiles,4):
        #     i+=1
        #     newImg = blend_two_images(infile1,infile2)
        #     newImg.save('/home/josh/Desktop/new_perfecto/perfecto__'+str(i)+'.jpg',
        #                 'JPEG')
        
