import torch
import torch.utils.data
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit_sequence/train/" ### add _sequence
        self.label_dir = cityscapes_meta_path + "/label_imgs/" ### this folder was created at my home with the code above ("preprocess_data.py")

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir

            file_names = sorted(os.listdir(train_img_dir_path)) ### sort the files alphanumerically
            file_names = file_names[19::30] ### get the "19th" frame of each sequence
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        gt_sisr = imread(img_path) # (shape: (1024, 2048, 3))

        label_img_path = example["label_img_path"]
        gt_sssr = imread(label_img_path) # (shape: (1024, 2048))

        ################################################################################################################
        # select a 512x512 random crop from the img and labels (gt_sssr and gt_sisr) and downsample the img to 256x256:
        ################################################################################################################
        start_x = np.random.randint(low=0, high=(self.img_w - 512))
        end_x = start_x + 512
        start_y = np.random.randint(low=0, high=(self.img_h - 512))
        end_y = start_y + 512

        gt_sisr = gt_sisr[start_y:end_y, start_x:end_x] # (shape: (512, 512, 3))
        gt_sssr = gt_sssr[start_y:end_y, start_x:end_x] # (shape: (512, 512))
        
        ################################################################################################################
        # flip the img and the label with 0.5 probability:
        ################################################################################################################
        #flip = np.random.randint(low=0, high=2)
        #if flip == 1:
        #    gt_sssr = cv2.flip(gt_sssr, 1)
        #    gt_sisr = cv2.flip(gt_sisr, 1)
        #
        ################################################################################################################
        # resize gt_sisr without interpolation to obtain img (the one used for training in both SSSR and SISR branches)
        ################################################################################################################
        img = resize(gt_sisr, (256, 256), preserve_range=True, anti_aliasing=True) # (shape: (256, 256, 3))
        #img = resize(gt_sisr, (self.new_img_h, self.new_img_w), preserve_range=True, anti_aliasing=True) # (shape: (512, 1024, 3))
            
        ################################################################################################################
        # normalize the img (with the mean and std):
        ################################################################################################################
        img = img/255.0
        mean = np.mean(img, axis=(0,1))
        std = np.std(img, axis=(0,1))
        img = img - mean
        img = img/std
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        gt_sisr = gt_sisr/255.0
        gt_sisr = gt_sisr - mean
        gt_sisr = gt_sisr/std
        gt_sisr = np.transpose(gt_sisr, (2, 0, 1)) # (shape: (3, 512, 512))
        gt_sisr = gt_sisr.astype(np.float32)
        
        gt_sssr = gt_sssr.astype(np.long)
        
        ################################################################################################################
        # convert numpy -> torch:
        ################################################################################################################
        img = torch.from_numpy(img) # (shape: (3, 256, 256))
        gt_sisr = torch.from_numpy(gt_sisr) # (shape: (3, 512, 512))
        gt_sssr = torch.from_numpy(gt_sssr) # (shape: (512, 512))
        
        data = {
            "img" : img,
            "mean" : mean,
            "std" : std,
            "gt_sisr" : gt_sisr,
            "gt_sssr" : gt_sssr
        }
        return data

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit_sequence/val/" ### add _sequence
        self.label_dir = cityscapes_meta_path + "/label_imgs/" ### this folder was created at my home with the code above ("preprocess_data.py")

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for val_dir in val_dirs:
            val_img_dir_path = self.img_dir + val_dir

            file_names = sorted(os.listdir(val_img_dir_path)) ### sort the files alphanumerically
            file_names = file_names[19::30] ### get the "19th" frame of each sequence
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = val_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        gt_sisr = imread(img_path) # (shape: (1024, 2048, 3))
        # resize gt_sisr without interpolation to obtain img (the one used for training in both SSSR and SISR branches)
        img = resize(gt_sisr, (self.new_img_h, self.new_img_w), preserve_range=True, anti_aliasing=True) # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        gt_sssr = imread(label_img_path) # (shape: (1024, 2048))

        ################################################################################################################
        # normalize the img (with the mean and std):
        ################################################################################################################
        img = img/255.0
        mean = np.mean(img, axis=(0,1))
        std = np.std(img, axis=(0,1))
        img = img - mean
        img = img/std
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)
        
        gt_sisr = gt_sisr/255.0
        gt_sisr = gt_sisr - mean
        gt_sisr = gt_sisr/std
        gt_sisr = np.transpose(gt_sisr, (2, 0, 1)) # (shape: (3, 1024, 2048))     
        gt_sisr = gt_sisr.astype(np.float32)
        
        gt_sssr = gt_sssr.astype(np.long)
        
        ################################################################################################################
        # convert numpy -> torch:
        ################################################################################################################
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))
        gt_sisr = torch.from_numpy(gt_sisr) # (shape: (3, 1024, 2048))
        gt_sssr = torch.from_numpy(gt_sssr) # (shape: (1024, 2048))
        
        data = {
            "img" : img,
            "mean" : mean,
            "std" : std,
            "gt_sisr" : gt_sisr,
            "gt_sssr" : gt_sssr,
            "img_id" : img_id
        }
        return data

    def __len__(self):
        return self.num_examples