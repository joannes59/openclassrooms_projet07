#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:09:57 2026

@author: joannes
"""

import os
import pandas as pd
import numpy as np
import cv2
import time
import logging

# Configuration de base : afficher les INFO dans la console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


COLUMNS=["path", "filename", "width", "height",
         "mode", "format", "size_bytes",  "bits", "label",
         "is_grayscale", "property", "brightness", "contrast", "sharpness",
          'dark_ratio', 'bright_ratio', 'median_intensity',
          'property', 'state'
]


class ImagePreprocess:
    """
    Classe pour la gestion des images avec un dataframe sauvegardé.
    """

    def __init__(self, 
                 filepath: str = "data/images.parquet", 
                 outpath: str = "preprocessed",
                 datapath: str = "data"):
        
        self.filepath = filepath
        self.datapath = datapath
        self.outpath = outpath
        self.df = pd.DataFrame()


    def load_dataframe(self) -> pd.DataFrame:
        """Charge le dataframe existant ou en crée un nouveau."""
        if os.path.exists(self.filepath):
            if self.filepath.endswith(".parquet"):
                self.df = pd.read_parquet(self.filepath)
            elif self.filepath.endswith(".csv"):
                self.df = pd.read_csv(self.filepath)
        else:
            self.df = pd.DataFrame(columns=COLUMNS)

        
    def save(self):
        """Sauvegarde le dataframe au format parquet ou csv."""
        if self.filepath.endswith(".parquet"):
            self.df.to_parquet(self.filepath, index=False)
        elif self.filepath.endswith(".csv"):
            self.df.to_csv(self.filepath, index=False)



    def update_label(self):
        """ Update un dataframe avec les labels pour le traitement d'image """
        start_time = time.time()
                
        for index, row in self.df.iterrows():
            
            if "cancer" in row["path"]:
                label = "cancer"
            elif "normal" in row["path"]:
                label = "normal"
            else:
                label = "None"
            
            self.df.loc[index, ['label']] = [label]
        
        end_time = round(time.time() - start_time, 3)
        logging.info(f" Labelized {self.df.shape[0]} images in {end_time} second")
            
        
    def preprocess_image(self, input_path, output_path):
    
        # lecture image (RGB en BGR dans OpenCV)
        img = cv2.imread(input_path)
    
        # conversion en grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # redimensionnement
        gray = cv2.resize(gray, (224, 224))
    
        # conversion vers 3 canaux pour encodeur
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
        # sauvegarde
        cv2.imwrite(output_path, rgb)
                
    def preprocess(self):
        """ Load path file image from dataframe and preprocess images """
        start_time = time.time()
        self.load_dataframe()
        
        # select images to preprocess
        out_path = os.path.join(self.datapath, self.outpath)
        os.makedirs(out_path, exist_ok=True)
        print('------outpath------', out_path)
        
        for index, row in self.df.iterrows():
            
            if row['state'] == "None":
                file_path_in = os.path.join(row["path"] , row["filename"])
                file_path_out = os.path.join(out_path, row["filename"])
                
                self.preprocess_image(file_path_in, file_path_out)
                
        end_time = round(time.time() - start_time, 3)
        logging.info(f"Image convertion images in {end_time} second") 
            
        
        
        

if __name__ == "__main__":
    
    manager = ImagePreprocess()
    manager.preprocess()
    

