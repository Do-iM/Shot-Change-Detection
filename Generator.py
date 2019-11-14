#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import os
import random

# functions declaration
def concat2image(img1, img2):
    img = Image.new('RGB', (224, 224))
    img.paste(img1.crop((0, 0, 112, 224)), (0, 0))
    img.paste(img2.crop((112, 0, 224, 224)), (112, 0))
    return img

def concat4image(img1, img2, img3, img4):
    img = Image.new('RGB', (224, 224))
    img.paste(img1.crop((0, 0, 56, 224)), (0, 0))
    img.paste(img2.crop((56, 0, 112, 224)), (56, 0))
    img.paste(img3.crop((112, 0, 168, 224)), (112, 0))
    img.paste(img4.crop((168, 0, 224, 224)), (168, 0))
    return img

def generate(set_size, dissolve_level = 25, dissolve_min_level = 12, fade_level = 5, fade_min_level = 1):
    os.mkdir("0_Non")
    os.mkdir("1_Cut")
    os.mkdir("2_Dissolve")
    os.mkdir("3_Fade")
    
    basedir = os.listdir('Base')
    image_size = (224, 224)
    
    for i in range(0, set_size):
        # select random base set
        dir_index = random.randrange(0, len(basedir))
        imgs = os.listdir('Base/' + basedir[dir_index])

        # select two continuous image / resize to (224, 224)
        img_index = random.randrange(0, len(imgs) - 1)
        img1 = Image.open('Base/' + basedir[dir_index] + '/' + imgs[img_index])
        img1 = img1.resize(image_size)
        img2 = Image.open('Base/' + basedir[dir_index] + '/' + imgs[img_index + 1])
        img2 = img2.resize(image_size)

        # merge img1, img2 to img / save img to 'Non/#.img'
        img = concat2image(img1, img2)
        img.save('0_Non/' + str(i) + '.jpg')
        
    for i in range(0, set_size):
        # select two different base set(set A, set B)
        dir_index = random.sample(range(0, len(basedir)), 2)
        imgs_a = os.listdir('Base/' + basedir[dir_index[0]])
        imgs_b = os.listdir('Base/' + basedir[dir_index[1]])

        # select two random image / resize to (224, 224)
        img_index_a = random.randrange(0, len(imgs_a))
        img1 = Image.open('Base/' + basedir[dir_index[0]] + '/' + imgs_a[img_index_a])
        img1 = img1.resize(image_size)
        img_index_b = random.randrange(0, len(imgs_b))
        img2 = Image.open('Base/' + basedir[dir_index[1]] + '/' + imgs_b[img_index_b])
        img2 = img2.resize(image_size)

        # merge img1, img2 to img / save img to 'Cut/#.img'
        img = concat2image(img1, img2)
        img.save('1_Cut/' + str(i) + '.jpg')
        
    for i in range(0, set_size):
        # select two different base set(set A, set B)
        dir_index = random.sample(range(0, len(basedir)), 2)
        imgs_a = os.listdir('Base/' + basedir[dir_index[0]])
        imgs_b = os.listdir('Base/' + basedir[dir_index[1]])

        # select two continuous image / resize to (224, 224)
        img_index_a = random.randrange(0, len(imgs_a) - 1)
        img1_a = Image.open('Base/' + basedir[dir_index[0]] + '/' + imgs_a[img_index_a])
        img1_a = img1_a.resize(image_size)
        img2_a = Image.open('Base/' + basedir[dir_index[0]] + '/' + imgs_a[img_index_a + 1])
        img2_a = img2_a.resize(image_size)
        img_index_b = random.randrange(0, len(imgs_b) - 1)
        img1_b = Image.open('Base/' + basedir[dir_index[1]] + '/' + imgs_b[img_index_b])
        img1_b = img1_b.resize(image_size)
        img2_b = Image.open('Base/' + basedir[dir_index[1]] + '/' + imgs_b[img_index_b + 1])
        img2_b = img2_b.resize(image_size)

        # make dissolve image
        # if level is low, img# is similar to img#_a
        # if level is high, img# is similar to img#_b
        level = random.randrange(dissolve_min_level, dissolve_level - dissolve_min_level)
        img1 = Image.blend(img1_a, img1_b, level / dissolve_level)
        img2 = Image.blend(img2_a, img2_b, (level + 1) / dissolve_level)

        # merge img1, img2 to img / save img to 'Dissolve/#.img'
        img = concat2image(img1, img2)
        img.save('2_Dissolve/' + str(i) + '.jpg')
    
    for i in range(0, set_size):
        # select random base set
        dir_index = random.randrange(0, len(basedir))
        imgs = os.listdir('Base/' + basedir[dir_index])

        # select two continuous image / resize to (224, 224)
        img_index = random.randrange(0, len(imgs) - 1)
        img1 = Image.open('Base/' + basedir[dir_index] + '/' + imgs[img_index])
        img1 = img1.resize(image_size)
        img2 = Image.open('Base/' + basedir[dir_index] + '/' + imgs[img_index + 1])
        img2 = img2.resize(image_size)

        # make dissolve image
        # if level is low, img# is similar to img#_a
        # if level is high, img# is similar to img#_b
        level = random.randrange(fade_min_level, fade_level - fade_min_level)
        # if random < 0.5, bw is black
        # otherwise, bw is white
        bw = Image.new("RGB", image_size, (255, 255, 255) if random.random() < 0.5 else (0, 0, 0))
        if random.random() < 0.5: # image to black or white        
            img1 = Image.blend(img1, bw, level / fade_level)
            img2 = Image.blend(img2, bw, (level + 1) / fade_level)
        else: # black or white to image
            img1 = Image.blend(bw, img1, level / fade_level)
            img2 = Image.blend(bw, img2, (level + 1) / fade_level)

        # merge img1, img2 to img / save img to 'Fade/#.img'
        img = concat2image(img1, img2)
        img.save('3_Fade/' + str(i) + '.jpg')
