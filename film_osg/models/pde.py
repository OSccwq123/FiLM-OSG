"""PDE training wrapper for FiLM-OSG reproducibility experiments.

Portions of this module are adapted from the DUE project:
https://github.com/AI4Equations/due
DUE is distributed under the LGPL-2.1 license. Local changes here keep the
training interface compatible with the reproduced OSG/FiLM-OSG scripts.
"""

import os
from numpy import savetxt
import matplotlib.pyplot as plt
from time import time
import random
import numpy as np
import torch

from ..utils import get_loss, get_optimizer, get_schedule


class PDE:
    """Basic training loop for PDE evolution models."""

    def __init__(self, trainX, trainY, network, config):
        super().__init__()

        self.trainX = torch.from_numpy(trainX)
        self.trainY = torch.from_numpy(trainY) 
        self.memory_steps = self.trainX.shape[-1]
        self.multi_steps = self.trainY.shape[-1]
        
        self.set_seed(config["seed"])
        self.device = config["device"]
        self.mynet = network.to(self.device)
        self.nepochs = config["epochs"]
        self.bsize   = config["batch_size"]
        
        self.lr      = config["learning_rate"]
        self.optimizer = get_optimizer(config["optimizer"], self.mynet, self.lr)
        self.scheduler = get_schedule(self.optimizer, config["scheduler"], self.nepochs, self.bsize, self.trainX.shape[0])
        self.verbose   = config["verbose"]
        
        self.loss_func = get_loss(config["loss"])
        self.save_path   = config["save_path"]
        os.makedirs(self.save_path, exist_ok=True)
        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.trainX, self.trainY), batch_size=self.bsize, shuffle=True)
        
    def train(self):
        self.summary()
        self.hist   = torch.zeros(self.nepochs,1)
        start = time()

        min_loss = 10000000000.0
        
        for ep in range(self.nepochs):
            self.mynet.train()
            train_step = 0
            for xx, yy in self.train_loader:
                xx = xx.to(self.device)
                yy = yy.to(self.device)
                
                pred = torch.zeros_like(yy)
                for t in range(self.multi_steps):
                    pred[...,t] = self.mynet(xx) #(batch_size, output_dim)
                    xx   = torch.cat((xx[...,1:], pred[...,t:t+1]), -1)
                
                loss       = self.loss_func(yy, pred)
                train_step += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler != None:
                    self.scheduler.step()
            train_step /= len(self.train_loader)
            if train_step < min_loss:
                torch.save(self.mynet, self.save_path+"/model")
                min_loss = train_step
            self.hist[ep,0] = train_step
            
            if (ep+1)%self.verbose ==0:
                end = time()
                print(f"Epoch {ep+1} --- Time: {end-start:.2f} seconds --- Training loss: {train_step}")
                start = end
                
    def save_hist(self, xlog=False, ylog=True):
        savetxt(self.save_path+"/training_history.csv", self.hist)
        
        plt.figure(figsize=(9,9))
        plt.plot(range(1,self.nepochs+1), self.hist[:,0], label="Train")
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.savefig(self.save_path+"/training_history.png")
        plt.close()

    def summary(self):
        """Print all trainable variables."""
        print("Number of trainable parameters:", self.mynet.count_params())
        print()
        print("Number of epochs:", self.nepochs)
        print("Batch size:", self.bsize)
        print("The model is trained on "+ self.device)
        
    def set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

