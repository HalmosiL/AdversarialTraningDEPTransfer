import os
import torch
import time
import sys
import glob

class DatasetAdversarial:    
    def __init__(self, con_conf_path, data_queue_path, slice_, mode_):
        self.con_conf_path = con_conf_path
        self.data_queue_path = data_queue_path
        self.slice_ = slice_
        self.mode_ = mode_
        self.bact_T_Queue = None
        self.epoch=0
        
    def __len__(self):
        return sys.maxsize
    
    def __addToEpoch__(self):
        self.epoch += 1

    def __getitem__(self, idx):        
        path_a = int(idx / self.slice_)
        path_b = idx % self.slice_

        if(self.mode_ == "train" and self.epoch == 0):
            image_normal_path = self.data_queue_path + "image_normal" + str(path_a) + "_" + str(path_b) + "_.pt"
            label_normal_path = self.data_queue_path + "label_normal" + str(path_a) + "_" + str(path_b) + "_.pt"
            image_adversarial_path = self.data_queue_path + "image_adversarial" + str(path_a) + "_" + str(path_b) + "_.pt"
            label_adversarial_path = self.data_queue_path + "label_adversarial" + str(path_a) + "_" + str(path_b) + "_.pt"

            if(
                os.path.exists(image_normal_path) and
                os.path.exists(label_normal_path) and
                os.path.exists(image_adversarial_path) and
                os.path.exists(label_adversarial_path)
            ):
                try:
                    image_normal = torch.load(image_normal_path).clone()
                    label_normal = torch.load(label_normal_path).clone()
                    image_adversarial = torch.load(image_adversarial_path).clone()
                    label_adversarial = torch.load(label_adversarial_path).clone()
                    
                    image = torch.cat((image_normal, image_adversarial), dim=0)
                    label = torch.cat((label_normal, label_adversarial), dim=0)
                    
                    return [
                        image.reshape(1, *image.shape),
                        label.reshape(1, *label.shape),
                        [image_normal_path, label_normal_path, image_adversarial_path, label_adversarial_path]
                    ]
                except Exception as e:
                    print("wrong")
                    return [[image_normal_path, label_normal_path, image_adversarial_path, label_adversarial_path]]

            return []
        elif(self.mode_ == "val" or self.epoch > 0):
            if(idx == 0):
                if(self.epoch % 2 == 0):
                    self.bact_T_Queue = glob.glob("../backupQueue2/image_*.pt")
                else:
                    self.bact_T_Queue = glob.glob("../backupQueue1/image_*.pt")
            
            image_path = self.data_queue_path + "image_" + str(path_a) + "_" + str(path_b) + "_.pt"
            label_path = self.data_queue_path + "label_" + str(path_a) + "_" + str(path_b) + "_.pt"

            if(
                os.path.exists(image_path) and
                os.path.exists(label_path)
            ):
                try:
                    image_ = torch.load(image_path).clone()
                    label_ = torch.load(label_path).clone()
                    
                    if(len(self.bact_T_Queue) != 0):
                        image_path_2 = self.bact_T_Queue[0]
                        label_path_2 = self.bact_T_Queue[0].replace("image", "label")
                        
                        print(image_path_2)
                        print(label_path_2)
                        
                        image_2 = torch.load(image_path_2).clone()
                        label_2 = torch.load(label_path_2).clone()
                    
                        image_ = torch.cat((image_, image_2), dim=0)
                        label_ = torch.cat((label_, label_2), dim=0)
                    
                        return [
                            image_.reshape(1, *image_.shape),
                            label_.reshape(1, *label_.shape),
                            [image_path, image_path_2, label_path, label_path_2]
                        ]
                    
                    return [
                        image_.reshape(1, *image_.shape),
                        label_.reshape(1, *label_.shape),
                        [image_path, label_path]
                    ]
                except Exception as e:
                    print(e)
                    print("wrong")
                    return [[image_path, label_path]]

            return []
