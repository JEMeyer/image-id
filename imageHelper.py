import os, sys, glob
from PIL import Image


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
    img = Image.open(imgFile)
    # I hate no case statements
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
        #Save normalized normal image
        abspath = '/home/joe/Projects/Python/CigarID/data/'
        #newImg.save(os.path.join(abspath, '__new__' + base + '.jpg'), 'JPEG')
        newImg = flip_image(infile, 'lr')
        # save new image
        newImg.save(os.path.join(abspath, '_lr_' + base + '.jpg'), 'JPEG')
        #newImg = convert_color_of_img(squareImg,'gray')
        newImg = flip_image(infile, 'tb')
        # save new image
        newImg.save(os.path.join(abspath, '_tb_' + base + '.jpg'), 'JPEG')
        #newImg = convert_color_of_img(squareImg,'gray')
        newImg = flip_image(infile, 'r90')
        # save new image
        newImg.save(os.path.join(abspath, '_r90_' + base + '.jpg'), 'JPEG')
        #newImg = convert_color_of_img(squareImg,'gray')
        newImg = flip_image(infile, 'r180')
        # save new image
        newImg.save(os.path.join(abspath, '_r180_' + base + '.jpg'), 'JPEG')
        #newImg = convert_color_of_img(squareImg,'gray')
        newImg = flip_image(infile, 'r270')
        # save new image
        newImg.save(os.path.join(abspath, '_r270_' + base + '.jpg'), 'JPEG')
        
