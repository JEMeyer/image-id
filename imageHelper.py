import os, sys, glob
from PIL import Image


def resize_img(imgFile,newSize):
    '''
    imgFile is the absolute path to an image, or an JpegImageFile
    newSize is max pixel W and H of new image
    '''
    if isinstance(imgFile, str):
        img = Image.open(imgFile)
    else:
        img = imgFile
    img.thumbnail(newSize, Image.ANTIALIAS)
    return img


def paste_img_on_background(imgFile, backgroundSize):
    '''
    imgFile is the absolute path to an image, or an JpegImageFile
    backgroundSize is the size of the background onto which
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
    imgFile is the absolute path to an image, or an JpegImageFile
    color is either 'RGB' or 'gray' or 'bw'
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


if __name__ == '__main__':
    dir = sys.argv[1]

    for infile in glob.glob(dir + '*.jpg'):
        # read in file and split path apart
        file, ext = os.path.splitext(infile)
        base = os.path.basename(file)
        abspath = os.path.dirname(file)

        # do relevant tranformations on image
        smallImg = resize_img(infile, newSize=(128,128))
        squareImg = paste_img_on_background(smallImg,backgroundSize=(128,128))
        newImg = convert_color_of_img(squareImg,'gray')

        # save new image
        newImg.save(os.path.join(abspath, '_new_' + base + '.jpg'), 'JPEG')
