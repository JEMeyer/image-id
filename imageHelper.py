import os, sys, glob
from PIL import Image

def resize_images_in_dir(dir, size = (128,128)):
    '''
    dir is directory of images
    size (optional) is max pixel W and H of generated images
    '''
    for infile in glob.glob(dir + '*.jpg'):
        file, ext = os.path.splitext(infile)
        base = os.path.basename(file)
        abspath = os.path.dirname(file)
        img = Image.open(infile)
        img.thumbnail(size, Image.ANTIALIAS)
        img.save(os.path.join(abspath, '_resized_' + base + '.jpg'), 'JPEG')
    
    
def paste_on_background(dir, size = (128,128)):
    ''' 
    dir is directory of foreground images
    size (optional) is an integer tuple for background size
    '''
    for infile in glob.glob(dir + '*.jpg'):
        file, ext = os.path.splitext(infile)
        base = os.path.basename(file)
        abspath = os.path.dirname(file)

        img = Image.open(infile) 
        img_w, img_h = img.size
        # (255,255,255) = white
        background = Image.new('RGB', size)
        bg_w, bg_h = background.size
        offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))
        background.paste(img, offset)
        background.save(os.path.join(abspath, '_normalized_' + base + '.jpg'), 
                        'JPEG')


def convert_color(dir, color='gray'):
    ''' 
    dir is directory of images
    color is either 'RGB' or 'gray' or 'bw'
    '''
    for infile in glob.glob(dir + '*.jpg'):
        file, ext = os.path.splitext(infile)
        base = os.path.basename(file)
        abspath = os.path.dirname(file)
        img = Image.open(infile)
        if color == 'gray':
            img = img.convert('L')
        elif color == 'RGB':
            img = img.convert('RGB')
        elif color == 'bw':
            img = img.convert('1')
        img.save(os.path.join(abspath, '_color_' + base + '.jpg'), 'JPEG')








