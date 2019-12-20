import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import KerasFunction as K

def make_demo_image(model, origin_dir, input_dir, output_dir):
    os.mkdir(output_dir)

    names = ['Non', 'Cut', 'Dissolve', 'Fade']
    values = [0.05, 0.15, 0.5, 0.3] # calculate

    non_value = [0.9, 0.01, 0.02, 0.07]
    cut_value = [0.03, 0.9, 0.04, 0.03]
    dissolve_value = [0.1, 0.05, 0.8, 0.05]
    fade_value = [0.04, 0.01, 0.1, 0.85]

    width = 1280

    figdpi = 100
    figwidth = width - 224 * 4
    figheight = 224


    origins = os.listdir(origin_dir)
    origins.sort()
    inputs = os.listdir(input_dir)
    inputs.sort()

    for i in range(len(inputs)):
        plt.figure(figsize=(figwidth/figdpi,figheight/figdpi), dpi=figdpi)
        plt.ylim([0,1])

        values = K.single_detection(model, os.path.join(input_dir, inputs[i]))

        color = ['blue', 'blue', 'blue', 'blue']
        color[np.argmax(values)] = 'red'

        plt.bar(names, values, width=0.6, color=color)
        plt.savefig("a.png")
        plt.close()

        a = Image.open("a.png")

        img1 = Image.open(os.path.join(origin_dir, inputs[i])) # same with inputs name

        img1 = img1.resize((width, int(img1.height * (width / img1.width))))

        img4 = Image.open(os.path.join(input_dir, inputs[i])) # inputs name

        img = Image.new('RGB', (img1.width, img1.height + 224))
        img.paste(img1, (0, 0))
        img.paste(img4, (0, img1.height))
        img.paste(a, (img4.width, img1.height))

        draw = ImageDraw.Draw(img)
        draw.rectangle(((0, img1.height), (224*4, img1.height + 224)), outline="white", width=3)
        draw.line([(224*1, img1.height), (224*1, img1.height + 224)], fill="white", width=3)
        draw.line([(224*2, img1.height), (224*2, img1.height + 224)], fill="white", width=3)
        draw.line([(224*3, img1.height), (224*3, img1.height + 224)], fill="white", width=3)
        draw.rectangle(((0, img1.height), (224, img1.height + 224)), outline="red", width=5)

        img.save(os.path.join(output_dir, inputs[i])) # same with inputs name
