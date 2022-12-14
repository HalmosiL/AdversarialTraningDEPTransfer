import sys
import glob
import os
import torch
import json
import subprocess
import time
import socket

sys.path.insert(0, "../")
from dataset.GetDatasetLoader import getDatasetLoader
from dataset.Dataset import SemData
from train_tools.Train import train
import util.Transforms as transform
from util.Comunication import Comunication

def start(CONFIG_PATH, script):
    bashCommand = [script, CONFIG_PATH]
    list_files = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')
        
    CONFIG_PATH = sys.argv[1]
    CONFIG = json.load(open(CONFIG_PATH, "r+"))
    
    print("Init com_conf...")
    start(CONFIG_PATH, "./start_com_server.sh")
    time.sleep(5)
    
    Comunication.tcp_socket = socket.create_connection((CONFIG["CONFMANAGER_HOST"], CONFIG["CONFMANAGER_PORT"]))
    
    if(os.path.exists("../backupQueue1")):
        models_in_cache = glob.glob("../backupQueue1/" + "*.pt")
        for m in models_in_cache:
            os.remove(m)
    else:
        os.mkdir("../backupQueue1")
        
    if(os.path.exists("../backupQueue2")):
        models_in_cache = glob.glob("../backupQueue2/" + "*.pt")
        for m in models_in_cache:
            os.remove(m)
    else:
        os.mkdir("../backupQueue2")
    
    print("Clear model cache...")
    models_in_cache = glob.glob(CONFIG["MODEL_CACHE"] + "*.pt")
    for m in models_in_cache:
        os.remove(m)

    train_loader_adversarial = getDatasetLoader(
        CONFIG_PATH,
        type_="train",
        num_workers=CONFIG["NUMBER_OF_WORKERS_DATALOADER"],
        pin_memory=CONFIG["PIN_MEMORY_ALLOWED_DATALOADER"]
    )
    val_loader_adversarial = getDatasetLoader(
        CONFIG_PATH,
        type_="val",
        num_workers=CONFIG["NUMBER_OF_WORKERS_DATALOADER"],
        pin_memory=CONFIG["PIN_MEMORY_ALLOWED_DATALOADER"]
    )

    args_dataset = CONFIG['DATASET']

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transform.Compose([
        transform.Crop([args_dataset["train_h"], args_dataset["train_w"]], crop_type='center', padding=mean, ignore_label=args_dataset["ignore_label"]),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    val_loader = torch.utils.data.DataLoader(
        dataset=SemData(
            split='val',
            data_root=CONFIG['DATA_PATH'],
            data_list=CONFIG['DATASET']['val_list'],
            transform=val_transform
        ),
        batch_size=16,
        num_workers=4,
        pin_memory=False
    )

    train(CONFIG_PATH, CONFIG, train_loader_adversarial, val_loader_adversarial, val_loader, start)
