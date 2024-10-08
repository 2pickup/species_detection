from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import rioxarray
import random
from PIL import Image
import torch

class dataset(Dataset):
    """ Dataset that loads images 1000px² images """
    def __init__(self, csv_path, pft_path, s2_path, slope_path, height_path, aspect_path,
        bands_s2, slope, bands_slope, height, bands_height, aspect, bands_aspect,
        img_shape, normalize, split, transform = None):
        pft_data = pd.read_csv(f'{csv_path}/{split}_Cropped_statistics.csv')
        
        self.pft_data = pft_data
        self.pft_path = pft_path
        self.shape = img_shape
        # self.bands_s1 = bands_s1
        self.bands_s2 = bands_s2
        self.bands_slope = bands_slope
        self.bands_height = bands_height
        self.bands_aspect = bands_aspect
        self.slope =slope
        self.height =height
        self.aspect =aspect
        self.normalize = normalize
        self.weights = pft_data['Ratio_no_NA_pft']
        self.x_off = pft_data['xoff'].values
        self.y_off = pft_data['yoff'].values
        self.height = pft_data['height'].values
        self.width = pft_data['width'].values
        # self.s1_path = s1_path
        self.s2_path = s2_path
        self.slope_path = slope_path
        self.height_path = height_path
        self.aspect_path = aspect_path
        # self.s1_mean = np.array([-22.14737272, -14.24342987, -19.8134118, -12.42150688])
        # self.s1_std = np.array([7.58778242, 6.52665157, 9.50614584, 7.13784733])
        # 10band +NDVI +NIRv
        self.s2_mean = np.array([422.25931886, 494.94366198, 504.12564235, 653.11047113, 867.12871099, 959.66867187, 1066.72228436, 1074.79540187, 1077.22829352, 746.93448359,1327.82640078,360.49261885])
        self.s2_std = np.array([229.33224792, 313.63051532, 440.11844939, 548.8114301, 783.28882692, 880.95723919, 998.75470881, 1006.81528059, 1064.29156912, 771.20939508,3085.52803497,487.9178381])
        # 10band
        # self.s2_mean = np.array([422.25931886, 494.94366198, 504.12564235, 653.11047113, 867.12871099, 959.66867187, 1066.72228436, 1074.79540187, 1077.22829352, 746.93448359])
        # self.s2_std = np.array([229.33224792, 313.63051532, 440.11844939, 548.8114301, 783.28882692, 880.95723919, 998.75470881, 1006.81528059, 1064.29156912, 771.20939508])
        self.split= split
        self.bin_ranges = [0,5,10,15,20,25,30,35,40,45,50]
        self.transform = transform

    def __len__(self):
        return len(self.pft_data)
    
    def labels_to_bins(self, labels):
        # Replace NaN with a value that falls outside the predefined bins, e.g., -999
        labels_cleaned = np.where(np.isnan(labels), -999, labels)
        bins = np.digitize(labels_cleaned, self.bin_ranges, right=False)
        bins = bins - 1  # Shift bins to start from 0
        bins = np.clip(bins, 0, len(self.bin_ranges) - 2)
        # Optionally, assign NaN values a unique bin index, e.g., -1 to indicate exclusion
        modified_bins = np.copy(bins)
        modified_bins[labels_cleaned == -999] = -1  # Modify the copied array
        return modified_bins
    
    def normalize_s2(self, img):
        return img.clip(0, 10000) * 0.0001 # multiply scale factor,normalize to 0-1
    
    # def normalize_s1(self, img):
        # return img.clip(-30, 0) / -30 # normalize to 0-1
    
    # def normalize_s1(self, img):
    #     """Normalize Sentinel-1 images using mean and std."""
    #     normalized_img = (img - self.s1_mean[:, None, None]) / self.s1_std[:, None, None]
    #     return normalized_img
    
    # def normalize_s2(self, img):
    #     """Normalize Sentinel-2 images using mean and std."""
    #     normalized_img = (img - self.s2_mean[:, None, None]) / self.s2_std[:, None, None]
    #     return normalized_img
    
    def normalize_slope(self, img):
        return img.clip(0, 90) / 90 #normalize to 0-1

    def normalize_height(self, img):
        return img.clip(0, 1165) / 1165 #normalize to 0-1
    
    def normalize_aspect(self, img):
        return img.clip(0, 360) / 360 #normalize to 0-1
    
    def cut_pft(self, img):
        return img.clip(0, 50)
     
    def pad_image(self, image, target_shape=(256, 256), pad_value=np.nan):
        """
        Pads an image with pad_value to the right and bottom to match the target shape.
        
        Parameters:
        - image: numpy array, the image to be padded.
        - target_shape: tuple, the desired shape (height, width) of the output image.
        - pad_value: value used to pad the image.
        
        Returns:
        - Padded image as a numpy array with the specified target shape.
        """
        # Calculate how much padding is needed
        padding_height = max(target_shape[0] - image.shape[1], 0)
        padding_width = max(target_shape[1] - image.shape[2], 0)
        
        # Pad the image
        if padding_height > 0 or padding_width > 0:
            padded_image = np.pad(image, 
                                ((0,0),(0, padding_height), (0, padding_width)), 
                                mode='constant', 
                                constant_values=pad_value)
        else:
            padded_image = image  # No padding needed
        
        return padded_image

    def __getitem__(self, item):
        row = self.pft_data.iloc[item]
        
        src_pft = rioxarray.open_rasterio(self.pft_path)
        # src_s1 = rioxarray.open_rasterio(self.s1_path)
        src_s2 = rioxarray.open_rasterio(self.s2_path)
        src_slope = rioxarray.open_rasterio(self.slope_path)
        src_height = rioxarray.open_rasterio(self.height_path)
        src_aspect = rioxarray.open_rasterio(self.aspect_path)
        
        no_pft = True
        
        while no_pft:
            x_off = self.x_off[item]
            y_off = self.y_off[item]
            if self.split in ['Test','Validation']:
                window = slice(y_off, y_off+self.width[item]), slice(x_off, x_off+self.width[item])
                label = src_pft.isel(band=0, y=window[0], x=window[1]).values

                if row['count'] != 0:
                    if np.all(label == 0):
                        no_pft = True
                    else:
                        no_pft = False
                else:
                    no_pft = False


                # image_s1 = src_s1.isel(band=[b - 1 for b in self.bands_s1] , y=window[0], x=window[1]).values.astype('float32')
                image_s2 = src_s2.isel(band=[b - 1 for b in self.bands_s2] , y=window[0], x=window[1]).values.astype('float32')
                image_slope = src_slope.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
                image_height = src_height.isel(band=[b - 1 for b in self.bands_height] , y=window[0], x=window[1]).values.astype('float32')
                image_aspect = src_aspect.isel(band=[b - 1 for b in self.bands_aspect] , y=window[0], x=window[1]).values.astype('float32')
                if self.height[item] != 256 or self.width[item] != 256:
                #if self.height[item] != 1000 or self.width[item] !=1000:
                    # image_s1 = self.pad_image(image_s1)
                    image_s2 = self.pad_image(image_s2)
                    image_slope = self.pad_image(image_slope)
                    image_height = self.pad_image(image_height)
                    image_aspect = self.pad_image(image_aspect)
            else:
                window_size = 256
                window = slice(y_off, y_off+window_size), slice(x_off, x_off+window_size)

                label = src_pft.isel(band=0, y=window[0], x=window[1]).values
                
                if row['count'] != 0:
                    if np.all(label == 0):
                        no_pft = True
                    else:
                        no_pft = False
                else:
                    no_pft = False

                ##print("4")

                image_s2 = src_s2.isel(band=[b - 1 for b in self.bands_s2] , y=window[0], x=window[1]).values.astype('float32')
                image_aspect = src_aspect.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
                image_height = src_height.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
                image_slope = src_slope.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')

                ##print("5")
                ##print("process going on")
                ## image_slope = src_slope.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
                
            if self.normalize:
                # image_s1 = self.normalize_s1(image_s1)
                image_s2 = self.normalize_s2(image_s2)
                image_slope = self.normalize_slope(image_slope)
                image_aspect = self.normalize_slope(image_aspect)

                image_height = self.normalize_slope(image_height)
                    
                # image_slope = self.normalize_slope(image_slope)
                label = self.cut_pft(label)

            if self.transform:
                image_s2 = torch.from_numpy(image_s2)
                image_slope = torch.from_numpy(image_slope)
                image_height = torch.from_numpy(image_height)
                image_aspect = torch.from_numpy(image_aspect)

                # Apply transformations
                image_s2 = self.transform(image_s2)
                image_slope = self.transform(image_slope)
                image_height = self.transform(image_height)
                image_aspect = self.transform(image_aspect)

            image_out = np.vstack([image_s2, image_slope,image_height, image_aspect])

                    
            #if self.slope:
            #    image_out = np.vstack([image_s2#,image_s1,image_slope
            #                           ])
            #else:
            #    image_out = np.vstack([image_s2#,image_s1
            #                           ])

            # if label is not None:
            #     bin_label = self.labels_to_bins(label)
            #     masks = (label != 0).astype(np.uint8)
        
            bin_label = self.labels_to_bins(label)

        return (image_out.astype(np.float32) , label, bin_label)
            # return (image_s2.astype(np.float32) , label, bin_label, masks)
            
            
        