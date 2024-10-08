import os
from tqdm import tqdm
import datetime
import pytz
from model import UNet
import torch
from utils2_species_focal_dem import get_dataloader, get_loss_function
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, confusion_matrix
import yaml 
import logging
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
# import plot_functions
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import rioxarray
from torch.utils.data import DataLoader
import torch.nn.functional as F


class dataset(Dataset):
    """ Dataset that loads images 256px² images """
    def __init__(self, csv_path, pft_path, s2_path, slope_path, height_path, aspect_path,
        bands_s2, slope, bands_slope, height, bands_height, aspect, bands_aspect,
        img_shape, normalize, split, transform = None):
        pft_data = pd.read_csv(f'/bess23/jooseo/experiment/2022_spec_dem/Temp_file_256/Cropped_statistics_with_window_with_overlap.csv')
        
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
        # self.weights = pft_data['Ratio_no_NA_pft']
        self.x_off = pft_data['xoff'].values
        self.y_off = pft_data['yoff'].values
        self.height = pft_data['height'].values
        self.width = pft_data['width'].values
        self.x_min_geo=pft_data["x_min_geo"].values
        self.y_max_geo=pft_data["y_max_geo"].values
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
        return img.clip(0, 10) 
     
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
        
        #no_pft = True
        x_min_out=self.x_min_geo[item]
        y_max_out=self.y_max_geo[item]
        
        #while no_pft:
        x_off = self.x_off[item]
        y_off = self.y_off[item]
            #if self.split in ['Test','Validation']:
        window = slice(y_off, y_off+self.width[item]), slice(x_off, x_off+self.width[item])
                #label = src_pft.isel(band=0, y=window[0], x=window[1]).values
                #no_pft = np.all(label == 0) if (row['count'] != 0) else False

                # image_s1 = src_s1.isel(band=[b - 1 for b in self.bands_s1] , y=window[0], x=window[1]).values.astype('float32')
        image_s2 = src_s2.isel(band=[b - 1 for b in self.bands_s2] , y=window[0], x=window[1]).values.astype('float32')
        image_slope = src_slope.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
        image_aspect = src_aspect.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
        image_height = src_height.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
        if self.height[item] != 256 or self.width[item] != 256:
                #if self.height[item] != 1000 or self.width[item] !=1000:
                    # image_s1 = self.pad_image(image_s1)
            image_s2 = self.pad_image(image_s2)
            image_slope = self.pad_image(image_slope)
            image_aspect = self.pad_image(image_aspect)
            image_height = self.pad_image(image_height)
        if self.normalize:
            image_s2 = self.normalize_s2(image_s2)
            image_slope = self.normalize_slope(image_slope)
            image_aspect = self.normalize_slope(image_aspect)
            image_height = self.normalize_slope(image_height)
            # label = self.cut_GEDI(label)
            
        image_out = np.vstack([image_s2, image_height,image_slope, image_aspect])
            
            
                    
            #else:
            #    window_size = 256
            #    window = slice(y_off, y_off+window_size), slice(x_off, x_off+window_size)

            #    label = src_pft.isel(band=0, y=window[0], x=window[1]).values
                
            #    if row['count'] != 0:
            #        if np.all(label == 0):
            #            no_pft = True
            #        else:
            #            no_pft = False
            #    else:
            #        no_pft = False


                ## code before
                #xoff, yoff = np.random.randint(0, self.width[item] - self.shape), np.random.randint(0, self.height[item] - self.shape)
                ##print("1")
                #window = slice(y_off+yoff, y_off+yoff+self.shape), saclice(x_off+xoff, x_off+xoff+self.shape)
                ##print("2")
                ##window = slice(y_off, y_off+self.shape), slice(x_off, x_off+self.shape)

                #label = src_pft.isel(band=0, y=window[0], x=window[1]).values
                ##print("3")
                ##no_pft = np.all(label == 0) if (row['count'] != 0) else False
                #if row['count'] != 0:
                #    if np.all(label == 0):
                #        no_pft = True
                #    else:
                #        sliced_tif = src_pft.isel(band=0, y=window[0], x=window[1])
                #        sliced_tif.rio.to_raster(f'/bess23/jooseo/testing/sliced_{item}.tif')
                #        print("slicing done1")
                #        no_pft = False
                #    
                #else:
                #    sliced_tif = src_pft.isel(band=0, y=window[0], x=window[1])
                #    sliced_tif.rio.to_raster(f'/bess23/jooseo/testing/sliced_{item}.tif')
                #    print("slicing done2")
                #    no_pft = False
                    
                
                ##print("4")

                ## image_s1 = src_s1.isel(band=[b - 1 for b in self.bands_s1] , y=window[0], x=window[1]).values.astype('float32')

                    

                ##print("5")
                ##print("process going on")
                ## image_slope = src_slope.isel(band=[b - 1 for b in self.bands_slope] , y=window[0], x=window[1]).values.astype('float32')
                
        #    if self.normalize:
                # image_s1 = self.normalize_s1(image_s1)
        #        image_s2 = self.normalize_s2(image_s2)
                    
                # image_slope = self.normalize_slope(image_slope)
        #        label = self.cut_pft(label)

                    
            #if self.slope:
            #    image_out = np.vstack([image_s2#,image_s1,image_slope
            #                           ])
            #else:
            #    image_out = np.vstack([image_s2#,image_s1
            #                           ])

            # if label is not None:
            #     bin_label = self.labels_to_bins(label)
            #     masks = (label != 0).astype(np.uint8)
        
            
        return (image_out.astype(np.float32) ,x_off,y_off,self.width[item],x_min_out,y_max_out)
            # return (image_s2.astype(np.float32) , label, bin_label, masks)
            

'''Loader'''
def get_dataloader(
        batch_size,
        csv_path, pft_path, s2_path, slope_path, height_path, aspect_path,
        bands_s2, slope, bands_slope, height, bands_height, aspect, bands_aspect,
        img_shape, normalize, split,  num_workers
        ):
    data = dataset(
        csv_path=csv_path, pft_path=pft_path,
        s2_path = s2_path, slope_path = slope_path , height_path = height_path , aspect_path = aspect_path,
        bands_s2=bands_s2,  slope = slope, bands_slope = bands_slope, height = height, bands_height = bands_height, aspect = aspect, bands_aspect = bands_aspect,
        img_shape=img_shape, normalize=normalize, split=split
        )
    epoch_size = len(data)
    if split =='Train':
        sampler = torch.utils.data.WeightedRandomSampler(
            data.weights.values,epoch_size
            )
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last = True
            )
    else:
        sampler = torch.utils.data.SequentialSampler(data)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last= True
            )
        
    data_loader = DataLoader(
        data,batch_sampler=batch_sampler, num_workers=num_workers
        )
    
    return(data_loader)

import rioxarray
import xarray as xr
from affine import Affine   
def save_prediction_to_geotiff(predicted_data, target_name,output_directory,x_off,y_off,width,x_min_out,y_max_out):  
    #target_path = f"{output_directory}/{target_name}.geotif"
    target_path = f"{output_directory}/{target_name}.tif"
        # Adjust the dimensions to remove 50 pixels from each edge
    cropped_predicted_data = predicted_data[50:-50, 50:-50]  # Assuming 2D array; adjust if 3D
    
    # Create an xarray DataArray with the cropped data
    da = xr.DataArray(cropped_predicted_data, dims=["y", "x"])
    
    # Adjust x_min_out and y_max_out for the cropping
    new_x_min_out = x_min_out + 50 * 10  # Move right by 50 pixels
    new_y_max_out = y_max_out - 50 * 10  # Move up by 50 pixels
    
    # Calculate the new transform
    # transform = [10, 0, new_x_min_out, 0, -10, new_y_max_out]
    transform = Affine.translation(new_x_min_out, new_y_max_out) * Affine.scale(10, -10)
    # Update the DataArray with the spatial information
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_crs("EPSG:32652", inplace=True)
    
    # Save the GeoTIFF
    da.rio.to_raster(target_path)
    try:
        os.chmod(target_path, 0o777)
    except:
        print('permission error to chmod')

def arg_parser():
    MODEL_TIME = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d_%Hh%M")
    
    with open('/bess23/jooseo/config/config_spec_dem.yaml') as f:
        cfg = yaml.safe_load(f)
    experiment_name =str(cfg['experiment_name'])
    year = cfg['year']
    data_root_dir = cfg['data_root_dir']
    #vrt_path_korea_S2 = os.path.join(data_root_dir,f'experiment/{experiment_name}/data/','korea_S2.vrt')
    vrt_path_korea_S2 = os.path.join('/bess23/jooseo/experiment',experiment_name,'data/cropped_S2.vrt')
    # vrt_path_korea_S1 = os.path.join(data_root_dir,f'experiment/{experiment_name}/data/','korea_S1.vrt')
    #vrt_path_korea_pft = os.path.join(data_root_dir,f'experiment/{experiment_name}/data/','korea_pft.vrt')
    vrt_path_korea_pft = os.path.join('/bess23/jooseo/experiment',experiment_name,'data/cropped_pft.vrt')
    # vrt_path_korea_LC = os.path.join(data_root_dir,f'experiment/{experiment_name}/data/','korea_LC.vrt')
    vrt_path_korea_slope = os.path.join(data_root_dir,f'experiment/{experiment_name}/data/','cropped_slope.vrt')
    vrt_path_korea_height = os.path.join(data_root_dir,f'experiment/{experiment_name}/data/','cropped_height.vrt')
    vrt_path_korea_aspect = os.path.join(data_root_dir,f'experiment/{experiment_name}/data/','cropped_aspect.vrt')
    processed_csv_path_copy = os.path.join(data_root_dir,'experiment',experiment_name,'0-final-preprocess_256')
    best_model_path = os.path.join(data_root_dir,'experiment',experiment_name,'best_model')
    root_path = os.path.join(data_root_dir,'experiment',experiment_name)
    ckpts_path = os.path.join(data_root_dir,'experiment',experiment_name,'checkpoint') 
    os.makedirs(best_model_path, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch Size during training') #from 64
    #parser.add_argument('--epoch', default=500, type=int, help='epoch to run')
    parser.add_argument('--num_worker', default=2, type=int, help='number of workers to load data')
    parser.add_argument('--csv_path', default=processed_csv_path_copy
                        , type=str, help='csv path')
    parser.add_argument('--pft_path', default=vrt_path_korea_pft
                        , type=str, help='pft path')
    #parser.add_argument('--s1_path', default=vrt_path_korea_S1
    #                    , type=str, help='s1 path')
    parser.add_argument('--s2_path', default=vrt_path_korea_S2
                        , type=str, help='s2 path')
    parser.add_argument('--slope_path', default=vrt_path_korea_slope
                        , type=str, help='slope path')
    parser.add_argument('--height_path', default=vrt_path_korea_height
                        , type=str, help='height path')
    parser.add_argument('--aspect_path', default=vrt_path_korea_aspect
                        , type=str, help='aspect path')                                                
    parser.add_argument('--model_path', default=best_model_path
                        , type=str, help='model save path')
    parser.add_argument('--root_path', default=root_path
                        , type=str, help='root path')
    parser.add_argument('--img_shape', default=256, type=int, help='input image size')
    parser.add_argument('--load_model', default=True, help='use pretrained model')
    parser.add_argument('--slope', default=True, help='use slope as input')
    parser.add_argument('--height', default=True, help='use height as input')
    parser.add_argument('--aspect', default=True, help='use aspect as input')
    parser.add_argument("--loss_function", type=str, choices=["MSE", "MAE", "SIG", "HUBER","CE"], 
                    required=True, help="Specify the loss function to use")
    parser.add_argument('--ckpts', default='/bess23/jooseo/experiment/2022_spec_dem/best_model/best_model_2024-10-02_13h22.pth', help='ckpts')
    parser.add_argument('--gpu', type=str, default='0,1', help='specify GPU devices')
    parser.add_argument('--model_time', type=str, default=MODEL_TIME, help='model training start time')
    parser.add_argument('--exp', type=str, default=experiment_name, help='experiment name')
    # imbalanced related
    # LDS
    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=2, help='LDS gaussian/laplace kernel sigma')
    # FDS
    parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
    parser.add_argument('--fds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=2, help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=50, help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=0, help='minimum(starting) bucket for FDS, 7 for NYUDv2')
    parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')
    # re-weighting: SQRT_INV / INV
    parser.add_argument('--reweight', type=str, default='none', choices=['none', 'inverse', 'sqrt_inv'],
                        help='cost-sensitive reweighting scheme')
    # two-stage training: RRT
    parser.add_argument('--retrain_fc', action='store_true', default=False,
                        help='whether to retrain last regression layer (regressor)')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained checkpoint file path to load backbone weights for RRT')
    return parser.parse_args()

def main(args,LOGGER):
    exp_name = args.exp
    if args.ckpts=="":
        LOGGER.info("*******THIS IS INFERENCE CODE, PUT --ckpts ARGS*******")
        exit()
    save_infer_path = os.path.join(os.path.dirname(args.ckpts),f'/bess23/jooseo/experiment/{args.exp}/Visualized_file/inferences/{args.exp}')
    os.makedirs(save_infer_path,exist_ok=True)
    try:
        os.chmod(save_infer_path, 0o777)
    except:
        print('permission error to chmod')
    # os.chmod(save_fig_path, 0o777)
    with open('/bess23/jooseo/config/config_spec_dem.yaml') as f:
        cfg = yaml.safe_load(f)
    experiment_name =str(cfg['experiment_name'])
    
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpts)))
    LOGGER.info(f'Test on the experiment: {exp_name}')
    DATALOADER_PARAMS = {
        'batch_size' : args.batch_size,
        'csv_path': args.csv_path,
        'pft_path' : args.pft_path, 
        's2_path' : args.s2_path,
        #'s1_path' : args.s1_path,
        'slope_path' : args.slope_path,
        'height_path' : args.height_path,
        'aspect_path' : args.aspect_path,
        #'bands_s1' : [1,2,3,4],   
        'slope' :args.slope,                
        'height' :args.height,
        'aspect' :args.aspect,
        'bands_slope' : [1],                     
        'bands_height' : [1],
        'bands_aspect' : [1],
        'img_shape' : args.img_shape, 
        # 'bands_s2' :[1,2,3,4,5,6,7,8,9,10,11,12], 
        'bands_s2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 수정필요
        'normalize' : True,
        'num_workers' : args.num_worker,  # CPU used to load the data
        
                     }                 
    #data loading
    #data loading
    # train_loader = get_dataloader(split = 'Train', **DATALOADER_PARAMS) 
    # val_loader = get_dataloader(split = 'Validation', **DATALOADER_PARAMS) 
    test_loader = get_dataloader(split = 'Test', **DATALOADER_PARAMS) 
    LOGGER.info(f'\ntest dataset: {len(test_loader.dataset)}')

    # Model
    n_channels =len(DATALOADER_PARAMS.get('bands_s2'))+len(DATALOADER_PARAMS.get('bands_slope'))+len(DATALOADER_PARAMS.get('bands_height'))+len(DATALOADER_PARAMS.get('bands_aspect'))

    n_classes = 9

    model = UNet(n_channels=n_channels, n_classes=n_classes,args= args)
    # model = UNet_deep(n_channels=n_channels, n_classes=n_classes,args= args)
    # model = UNet_huge(n_channels=n_channels, n_classes=n_classes,args= args)
    # model = UNet_large(n_channels=n_channels, n_classes=n_classes,args= args)
    model = model.cuda()
    torch.cuda.empty_cache() # Reduce model size

    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict

    # import torch.nn as nn
    # model = nn.DataParallel(model)
    if args.load_model:
        checkpoint_path=args.ckpts
        checkpoint = torch.load(checkpoint_path)

        # Load the model and optimizer states
        state_dict = remove_module_prefix(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict)
        # Load the model and optimizer states
        # model.load_state_dict(checkpoint['model_state_dict'])
        best_val_f1 = checkpoint['val_f1']
        epoch_saved = checkpoint['epoch']
        LOGGER.info(f'Loaded model checkpoint from: {checkpoint_path} \nBest validation accuracy: {best_val_f1:0.4f}\nSaved at epoch {epoch_saved}')
    import torch.nn as nn
    model = nn.DataParallel(model)
    torch.cuda.empty_cache() # Reduce model size
    # LOGGER.info(f"{torch.cuda.memory_summary(device=None, abbreviated=False)}")

    for epoch in range(1):

        LOGGER.info(f"------------current epoch : {epoch+1}------------")
       
        # =========== Test part ===========
        
        model.eval()
        # for visualization
        
        loop = tqdm(test_loader)
        for batch_idx,(images,x_off,y_off,width,x_min_out,y_max_out) in enumerate(loop):
        # for i in range(len(data)):
            # image, label = data.__getitem__(batch_idx)
            # images = np.expand_dims(images,0)
            images = images.cuda().float()
            images = torch.nan_to_num(images) # to remove nans
            out = model(images)

            probs = F.softmax(out, dim=1) 
            conf, predictions = torch.max(probs, 1)
            below_threshold_mask = (conf < 0.5) # Mask for confidence < 50%
            predictions[below_threshold_mask] = 9

            # Convert predictions to CPU and numpy for processing
            predictions_np = predictions.cpu().detach().numpy()
            torch.cuda.empty_cache() # Reduce model size

            # Iterate over each image in the batch
            for i in range(predictions_np.shape[0]):
                single_prediction = predictions_np[i]
                # Assuming single channel output, remove if model outputs multichannel
                if single_prediction.ndim > 2:
                    single_prediction = single_prediction[0]
                
                # Construct unique target name for each image
                target_name = f"image_{batch_idx * args.batch_size + i}"
                
                # Extract metadata for the current image in batch
                
                current_x_off = x_off[i].item()
                current_y_off = y_off[i].item()
                current_width = width[i].item()
                current_x_min_out = x_min_out[i].item()
                current_y_max_out = y_max_out[i].item()

                # Save the single image prediction
                save_prediction_to_geotiff(single_prediction, target_name, save_infer_path, current_x_off, current_y_off, current_width, current_x_min_out, current_y_max_out)
                
                # Clear memory
                del single_prediction
            # Clear memory
            del predictions,images,x_off,y_off,width,x_min_out,y_max_out, current_x_off, current_y_off, current_width, current_x_min_out, current_y_max_out, conf, probs, out
            torch.cuda.empty_cache() # Reduce model size



if __name__ == "__main__":
    
    args = arg_parser()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "1"

    cuda_available = torch.cuda.is_available()

    print(f"CUDA available: {cuda_available}")

    # If CUDA is available, print the version and the name of the current GPU
    if cuda_available:
        cuda_version = torch.version.cuda
        current_gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"CUDA Version: {cuda_version}")
        print(f"Current GPU: {current_gpu}")
    else:
        print("CUDA is not available. Check your system's GPU compatibility and PyTorch installation.")

    # makelog
    LOGGER = logging.getLogger()

    # 로그의 출력 기준 설정
    LOGGER.setLevel(logging.INFO)

    #log 출력 형식
    formatter = logging.Formatter("%(message)s")

    # log를 console에 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    main(args,LOGGER)