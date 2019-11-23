#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import os
import VGGModel

class mix2image:
    length = 6
    index = [0, 5]
    change = 3
    vgg = VGGModel.SingleVGG
    def gen(imgs):
        img = Image.new('RGB', (224, 224))
        img.paste(imgs[0].crop((0, 0, 112, 224)), (0, 0))
        img.paste(imgs[1].crop((112, 0, 224, 224)), (112, 0))
        return img

class mix4image:
    length = 10
    index = [0, 4, 5, 9]
    change = 5
    vgg = VGGModel.SingleVGG
    def gen(imgs):
        img = Image.new('RGB', (224, 224))
        img.paste(imgs[0].crop((0, 0, 56, 224)), (0, 0))
        img.paste(imgs[1].crop((56, 0, 112, 224)), (56, 0))
        img.paste(imgs[2].crop((112, 0, 168, 224)), (112, 0))
        img.paste(imgs[3].crop((168, 0, 224, 224)), (168, 0))
        return img

class concat2image:
    length = 2
    index = [0, 1]
    change = 1
    vgg = VGGModel.DoubleVGG
    def gen(imgs):
        img = Image.new('RGB', (448, 224))
        img.paste(imgs[0].crop((0, 0, 224, 224)), (0, 0))
        img.paste(imgs[1].crop((0, 0, 224, 224)), (224, 0))
        return img
