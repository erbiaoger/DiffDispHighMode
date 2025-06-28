# %%
from PIL import Image
import pathlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import shutil

def chang2RGB(image, filename, mode="RGB"):
    if mode == "RGB":
        # 将RGBA图像转换为RGB格式
        rgb_image = image.convert("RGB")
        rgb_image.save(filename)  # 保存为RGB格式的PNG图片
    elif mode == "L":
        # 将RGBA图像转换为灰度图
        gray_image = image.convert("L")
        gray_image.save(filename) # 保存为灰度图
    
    # print(filename.stem + " has been saved.")

def process_image(file): 
    try: 
        image = Image.open(file)
        chang2RGB(image, file, mode="L")
    except:
        # 如果转换失败，则删除该文件
        # os.remove(file)
        print(file)



files_rgb = pathlib.Path("dataset/demultiple/data_train/data").rglob("*.png")
files_l = pathlib.Path("dataset/demultiple/data_train/labels").rglob("*.png")

with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, sorted(files_rgb)), total=len(sorted(files_rgb)), desc="Processing RGB Images"))
    list(tqdm(executor.map(process_image, sorted(files_l)), total=len(sorted(files_l)), desc="Processing Label Images"))
