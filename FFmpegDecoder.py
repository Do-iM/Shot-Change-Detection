#!/usr/bin/env python
# coding: utf-8

import subprocess
import os

# video_to_image("sample.mp4", 'out_images', 60)
def video_to_image(video_name, output_dir, fps=30):
    os.mkdir(output_dir)
    
    ffmpeg_path = os.path.join("ffmpeg", "bin", "ffmpeg")
    output_path = os.path.join(output_dir, "%04d.jpg")
    
    subprocess.run([ffmpeg_path, "-i", video_name, "-vf", ''.join(["fps=", str(fps)]), output_path, "-hide_banner"], shell=True, check=True)
