import torch
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
IMAGE_SIZE = 384

class JPEGCompression:
    def __init__(self, quality_Range=(40,90)):
        self.quality = quality_Range

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        output = io.BytesIO()
        quality = random.randint(*self.quality)
        img.save(output, format="JPEG", quality=quality)
        output.seek(0)
        compressed_img = Image.open(output)
        return compressed_img


class Dataset_Loader(Dataset):
    def __init__(self, txt_path, train_flag=True,preprocess_mode = 0):
        super().__init__()
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        self.jpeg_compression = JPEGCompression()
        self.preprocess_mode = preprocess_mode
        mean = (0.485, 0.456, 0.406)
        aa_params = dict(
            translate_const=int(512 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )

        self.train_tf = transforms.Compose([
                #transforms.Resize(IMAGE_SIZE),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.01, 2.0))], 0.5),
                transforms.RandomApply([self.jpeg_compression], 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                #transforms.RandomRotation((-30,30)),
                #RandomResizedCropAndInterpolation((IMAGE_SIZE, IMAGE_SIZE), (0.6, 1.0), (3. / 4., 4. / 3.), 'random'),
                #rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params),

                #transforms.ColorJitter(brightness=.2, hue=.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


            ])

        # todo
        """
        timm::RandomResizedCropAndInterpolation
        RandomHorizontalFlip
        RandomVerticalFlip
        timm::rand_augment_transform / augment_and_mix_transform / auto_augment_transform
        timm::JPEGCompression
        """

        self.val_tf = transforms.Compose([
                #transforms.Resize(IMAGE_SIZE),
                #self.jpeg_compression,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.transform_resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.transform_crop = transforms.CenterCrop((IMAGE_SIZE,IMAGE_SIZE))
        self.transform_resize_crop = transforms.RandomResizedCrop((IMAGE_SIZE,IMAGE_SIZE))
        
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        del_list = []
        for i in range(len(imgs_info)):
            if len(imgs_info[i]) != 2:
                del_list.append(i)
        for idx in del_list:
            del imgs_info[idx]
        return imgs_info
     
    def padding_black(self, img):

        w, h  = img.size

        scale = IMAGE_SIZE / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = IMAGE_SIZE

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img
        
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')



        if self.preprocess_mode == 1:
            img = self.transform_resize(img)
        elif self.preprocess_mode ==2:
            if min(img.height,img.width) < IMAGE_SIZE:
                #print("padding")
                new_height =  int((img.height-IMAGE_SIZE)/2)+1
                new_width =   int((img.width - IMAGE_SIZE) / 2) + 1
                new_width = new_width if new_width >= 0  else 0
                new_height = new_height if new_height >= 0 else 0
                transforms.Pad((new_width,new_height))(img)

            img = self.transform_crop(img)
        else:
            rand_num = random.randint(1,3)
            if rand_num == 1:
                img = self.transform_resize(img)
            elif rand_num ==2:
                img = self.transform_resize_crop(img)
            else:
                if min(img.height,img.width) < IMAGE_SIZE:
                    #print("padding")
                    new_height =  int((img.height-IMAGE_SIZE)/2)+1
                    new_width =   int((img.width - IMAGE_SIZE) / 2) + 1
                    new_width = new_width if new_width >= 0  else 0
                    new_height = new_height if new_height >= 0 else 0
                    transforms.Pad((new_width,new_height))(img)

                img = self.transform_crop(img)
        #img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label
 
    def __len__(self):
        return len(self.imgs_info)




class DatasetLoaderTest(Dataset):
    def __init__(self, txt_path, train_flag=True, preprocess_mode=0):
        super().__init__()
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        self.jpeg_compression = JPEGCompression()
        self.preprocess_mode = preprocess_mode
        mean = (0.485, 0.456, 0.406)


        self.train_tf = transforms.Compose([
            # transforms.Resize(IMAGE_SIZE),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.01, 2.0))], 0.5),
            transforms.RandomApply([self.jpeg_compression], 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation((-30,30)),
            # RandomResizedCropAndInterpolation((IMAGE_SIZE, IMAGE_SIZE), (0.6, 1.0), (3. / 4., 4. / 3.), 'random'),
            # rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params),

            # transforms.ColorJitter(brightness=.2, hue=.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

        # todo
        """
        timm::RandomResizedCropAndInterpolation
        RandomHorizontalFlip
        RandomVerticalFlip
        timm::rand_augment_transform / augment_and_mix_transform / auto_augment_transform
        timm::JPEGCompression
        """

        self.val_tf = transforms.Compose([
            # transforms.Resize(IMAGE_SIZE),
            # self.jpeg_compression,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.transform_crop = transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))
        self.transform_resize_crop = transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE))

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        del_list = []
        for i in range(len(imgs_info)):
            if len(imgs_info[i]) != 2:
                del_list.append(i)
        for idx in del_list:
            del imgs_info[idx]
        return imgs_info

    def padding_black(self, img):

        w, h = img.size

        scale = IMAGE_SIZE / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = IMAGE_SIZE

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')

        if self.preprocess_mode == 1:
            img = self.transform_resize(img)
        elif self.preprocess_mode == 2:
            if min(img.height, img.width) < IMAGE_SIZE:
                # print("padding")
                new_height = int((img.height - IMAGE_SIZE) / 2) + 1
                new_width = int((img.width - IMAGE_SIZE) / 2) + 1
                new_width = new_width if new_width >= 0 else 0
                new_height = new_height if new_height >= 0 else 0
                transforms.Pad((new_width, new_height))(img)

            img = self.transform_crop(img)
        else:
            rand_num = random.randint(1, 3)
            if rand_num == 1:
                img = self.transform_resize(img)
            elif rand_num == 2:
                img = self.transform_resize_crop(img)
            else:
                if min(img.height, img.width) < IMAGE_SIZE:
                    # print("padding")
                    new_height = int((img.height - IMAGE_SIZE) / 2) + 1
                    new_width = int((img.width - IMAGE_SIZE) / 2) + 1
                    new_width = new_width if new_width >= 0 else 0
                    new_height = new_height if new_height >= 0 else 0
                    transforms.Pad((new_width, new_height))(img)

                img = self.transform_crop(img)
        # img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label ,img_path

    def __len__(self):
        return len(self.imgs_info)

if __name__ == "__main__":
    train_dataset = Dataset_Loader("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label)