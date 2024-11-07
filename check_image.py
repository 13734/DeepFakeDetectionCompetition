import  os
from  PIL import Image
from tqdm import tqdm



def check_path(path):

    with open(path, "r", encoding="utf-8") as f:
        list_all = f.read().split("\n")
    new_list = []
    for item in tqdm(list_all):
        image_path = item.split("\t")[0]
        img = Image.open(image_path)
        if img.width > 64 and img.height > 64:
            new_list.append(list_all)
        else:
            print(f"error{image_path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_list))

if __name__ == '__main__':
    check_path("train_data/result.txt")