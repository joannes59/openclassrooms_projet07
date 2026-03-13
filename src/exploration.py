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
          'dark_ratio', 'bright_ratio', 'median_intensity'
]


class ImageExploration:
    """
    Classe pour la gestion des images avec un dataframe sauvegardé.
    """

    def __init__(self, filepath: str = "data/images.parquet", datapath: str = "data"):
        self.filepath = filepath
        self.datapath = datapath
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

    def create_dataframe(self):
        """ Créer un nouveau dataframe au format parquet ou csv.
        avec la liste des fichiers images
        """
        start_time = time.time()
        
        file_paths = self.get_file_paths()
    
        if file_paths.empty:
            self.df = pd.DataFrame(columns=COLUMNS)
        else:
            
            data = []
        
            for _, row in file_paths.iterrows():
                
                data.append(
                    {
                        "path": row["path"],
                        "filename": row["filename"],
                        "format": row["format"],
                        "width": 0,
                        "height": 0,
                        "mode": "None",
                        "size_bytes": row["size_bytes"],
                        "is_grayscale": False,
                        "brightness": 0,
                        "contrast": 0,
                        "sharpness":0,
                        'dark_ratio': 0,
                        'bright_ratio': 0,
                        'median_intensity': 0,
                        'property': 'todo',
                        "bits": 0,
                        "label": "None",
                        "state": "None",
                    }
                )
        
            self.df = pd.DataFrame(data)
            
        end_time = round(time.time() - start_time, 3)
        logging.info(f" List {self.df.shape[0]} images in {end_time} second")

        
    def save(self):
        """Sauvegarde le dataframe au format parquet ou csv."""
        if self.filepath.endswith(".parquet"):
            self.df.to_parquet(self.filepath, index=False)
        elif self.filepath.endswith(".csv"):
            self.df.to_csv(self.filepath, index=False)


    def get_file_paths(self):
        """
        Parcourt tous les dossiers dans data et enregistre les chemins des fichiers.
        Returns:
            DataFrame avec les colonnes 'path', 'filename', 'format', 'size_bytes'
        """
        file_paths = []

        for root, dirs, files in os.walk(self.datapath):
            for filename in files:
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
                ):
                    file_path = os.path.join(root , filename)
                    size_bytes = int(os.path.getsize(file_path))
                    img_format = filename.lower().split('.')[-1]
                    file_paths.append({"path": root, 
                                       "filename": filename,
                                       "size_bytes": size_bytes,
                                       "format": img_format})

        return pd.DataFrame(file_paths)


    def update_img_property(self):
        """Crée un dataframe avec les propriétés pour le traitement d'image """
        start_time = time.time()
                
        for index, row in self.df.iterrows():
    
            try:
                file_path = os.path.join(row["path"] , row["filename"])
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
                if img is None:
                    continue
    
                # dimensions
                height, width = img.shape[:2]
    
                # nombre de canaux
                
                is_grayscale = False
                
                if len(img.shape) == 2:
                    mode = "L" 
                    is_grayscale = True
                    gray = img
                else:
                    channels = img.shape[2]
                    is_grayscale = (
                                    (img[:,:,0] == img[:,:,1]).all() and
                                    (img[:,:,0] == img[:,:,2]).all()
                                    )
    
                    if channels == 3:
                        mode = "RGB"
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    elif channels == 4:
                        mode = "RGBA"
                        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    else:
                        continue
    
                bits = img.dtype.itemsize * 8
                max_color = int(2^bits)
                
                brightness = int(gray.mean())
                median_intensity = int(np.median(gray))
                contrast = int(gray.std())
                sharpness = int(cv2.Laplacian(gray, cv2.CV_64F).var())
                dark_ratio = int(100.0 * np.mean(gray < int(0.15 * max_color)))
                bright_ratio = int(100.0 * np.mean(gray > int(0.85 * max_color)))
    

                self.df.loc[index, ['width', 'height', 'mode', 'bits', 'is_grayscale', 'brightness', 
                                    'contrast', 'sharpness', 'dark_ratio', 'bright_ratio', 'median_intensity', 'property'],
                            ] = [width, height, mode, bits, is_grayscale, brightness,
                                 contrast, sharpness, dark_ratio, bright_ratio, median_intensity, 'done']
                
            except:
                pass
            
        end_time = round(time.time() - start_time, 3)
        logging.info(f" Update {self.df.shape[0]} images properties in {end_time} second")


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
            
                
    def update_outliers(self):
        """ Update un dataframe avec les outliers """
        start_time = time.time()
        
        # Filter
        brightness = (self.df['brightness'] >= 98)
        contrast = (self.df['contrast'] >= 90)
        sharpness = (self.df['sharpness'] > 500)
        dark_ratio = (self.df['dark_ratio'] > 55)
        bright_ratio = (self.df['bright_ratio'] < 31)
        median_intensity = (self.df['median_intensity'] > 125)
        
        filter_img = (sharpness | brightness | contrast | dark_ratio | bright_ratio | median_intensity)
        
        self.df.loc[filter_img, ['state']] = "outlier"
        end_time = round(time.time() - start_time, 3)
        logging.info(f" Add filter {self.df.shape[0]} images in {end_time} second")

if __name__ == "__main__":
    manager = ImageExploration()
    manager.create_dataframe()
    manager.update_img_property()
    manager.update_label()
    manager.update_outliers()
    manager.save()
    

