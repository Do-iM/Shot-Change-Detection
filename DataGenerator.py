#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import os
import random
    
def frame_to_input(gen_f, frame_dir, input_dir):
    os.mkdir(input_dir)
    frames = os.listdir(frame_dir)
    for i in range(len(frames) - (gen_f.length - 1)):
        img_set = []
        for f_index in gen_f.index:
            img = Image.open(os.path.join(frame_dir, frames[i + f_index]))
            img = img.resize((224, 224))
            img_set.append(img)
        gen_f.gen(img_set).save(os.path.join(input_dir, frames[i]))

def generate(gen_f, set_size, base_dir, out_dir, dissolve_level = [10, 10], fade_level = [10, 10]):
    os.mkdir(out_dir)
    
    os.mkdir(os.path.join(out_dir, "0_Non"))
    os.mkdir(os.path.join(out_dir, "1_Cut"))
    os.mkdir(os.path.join(out_dir, "2_Dissolve"))
    os.mkdir(os.path.join(out_dir, "3_Fade"))
    
    basedir = os.listdir(base_dir)
    image_size = (224, 224)
    
    # Non
    for i in range(0, set_size):
        # select random base set
        while True:
            dir_index = random.randrange(0, len(basedir))
            imgs = os.listdir(os.path.join(base_dir, basedir[dir_index]))
            if len(imgs) >= gen_f.length:
                break

        # select n continuous image / resize to (224, 224)
        img_index = random.randrange(0, len(imgs) - (gen_f.length - 1))
        img_set = []
        for f_index in gen_f.index:
            img = Image.open(os.path.join(base_dir, basedir[dir_index], imgs[img_index + f_index]))
            img = img.resize(image_size)
            img_set.append(img)

        # merge imgs / save to 'Non/#.img'
        gen_f.gen(img_set).save(os.path.join(out_dir, '0_Non', str(i) + '.jpg'))
        
    # Cut
    for i in range(0, set_size):
        # select two different base set(set A, set B)
        while True:
            dir_index = random.sample(range(0, len(basedir)), 2)
            imgs_a = os.listdir(os.path.join(base_dir, basedir[dir_index[0]]))
            imgs_b = os.listdir(os.path.join(base_dir, basedir[dir_index[1]]))
            if len(imgs_a) >= gen_f.change and len(imgs_b) >= gen_f.length - gen_f.change:
                break

        # select two random image / resize to (224, 224)
        img_index_a = random.randrange(0, len(imgs_a) - (gen_f.change - 1))
        img_index_b = random.randrange(0, len(imgs_b) - (gen_f.length - gen_f.change - 1))
        img_set = []
        for f_index in gen_f.index:
            if f_index < gen_f.change:
                img = Image.open(os.path.join(base_dir, basedir[dir_index[0]], imgs_a[img_index_a + f_index]))
            else:
                img = Image.open(os.path.join(base_dir, basedir[dir_index[1]], imgs_b[img_index_b + f_index - gen_f.change]))
            img = img.resize(image_size)
            img_set.append(img)

        # merge imgs / save to 'Cut/#.img'
        gen_f.gen(img_set).save(os.path.join(out_dir, '1_Cut', str(i) + '.jpg'))
        
    # Dissolve
    for i in range(0, set_size):
        # select two different base set(set A, set B)
        while True:
            dir_index = random.sample(range(0, len(basedir)), 2)
            imgs_a = os.listdir(os.path.join(base_dir, basedir[dir_index[0]]))
            imgs_b = os.listdir(os.path.join(base_dir, basedir[dir_index[1]]))
            if len(imgs_a) >= gen_f.length and len(imgs_b) >= gen_f.length:
                break

        # select two continuous image / resize to (224, 224)
        img_index_a = random.randrange(0, len(imgs_a) - (gen_f.length - 1))
        img_index_b = random.randrange(0, len(imgs_b) - (gen_f.length - 1))
        img_set = []
        levels = random.randrange(dissolve_level[0], dissolve_level[1] + 1)
        for f_index in gen_f.index:
            img_a = Image.open(os.path.join(base_dir, basedir[dir_index[0]], imgs_a[img_index_a + f_index]))
            img_a = img_a.resize(image_size)
            img_b = Image.open(os.path.join(base_dir, basedir[dir_index[1]], imgs_b[img_index_b + f_index]))
            img_b = img_b.resize(image_size)
            img = Image.blend(img_a, img_b, (levels / 2 - gen_f.change + f_index) / levels)
            img_set.append(img)

        # merge imgs / save to 'Dissolve/#.img'
        gen_f.gen(img_set).save(os.path.join(out_dir, '2_Dissolve', str(i) + '.jpg'))
    
    # Fade
    for i in range(0, set_size):
        # select random base set
        while True:
            dir_index = random.randrange(0, len(basedir))
            imgs = os.listdir(os.path.join(base_dir, basedir[dir_index]))
            if len(imgs) >= gen_f.length:
                break

        # select two continuous image / resize to (224, 224)
        img_index = random.randrange(0, len(imgs) - (gen_f.length - 1))
        img_set = []
        levels = random.randrange(fade_level[0], fade_level[1] + 1)
        #bw = Image.new("RGB", image_size, (255, 255, 255) if random.random() < 0.5 else (0, 0, 0))
        bw = Image.new("RGB", image_size, (0, 0, 0))
        bw_switch = random.random() < 0.5
        for f_index in gen_f.index:
            img_a = Image.open(os.path.join(base_dir, basedir[dir_index], imgs[img_index + f_index]))
            img_a = img_a.resize(image_size)
            img_b = bw
            if bw_switch:
                (img_a, img_b) = (img_b, img_a)
            img = Image.blend(img_a, img_b, (levels / 2 - gen_f.change + f_index) / levels)
            img_set.append(img)

        # merge imgs / save to 'Fade/#.img'
        gen_f.gen(img_set).save(os.path.join(out_dir, '3_Fade', str(i) + '.jpg'))
