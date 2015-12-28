import os, sys, glob
from PIL import Image
  
def resize_images_in_dir(dir, size = (128,128)):
    '''
    dir is directory of images
    size (optional) is max pixel W and H of generated images
    '''
    for infile in glob.glob(dir + "*.jpg"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(file + "_.jpg", "JPEG")
        
def paste_on_background(dir, size = (128, 128)):
    ''' 
    dir is directory of foreground images
    size (optional) is an integer tuple for background size
    '''
    for infile in glob.glob(dir + "*.jpg"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile) 
        img_w, img_h = im.size
        background = Image.new('RGB', size)
        bg_w, bg_h = background.size
        offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))
        background.paste(im, offset)
        background.save(file + ".normalized.jpg", "JPEG")