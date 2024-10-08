# from dataset import dataset
from dataset_3_dem import dataset
import torch
from torch.utils.data import DataLoader
import numpy as np

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

'''LOSS'''
def get_loss_function(name):
    if name == "MSE":
        def MSE(out, target):
            """Mean Squarred error, calculated only where there is a gedi footprint ie. target!=0 """
            out = out.flatten()
            target = target.flatten()
            out = out[target !=0]
            target = target[target != 0]
            loss = ((target - out)**2).mean()
            return(loss)
        return MSE
    elif name == "MAE":
        def MAE(out,target):
            """Mean absolute error , calculated only where there is a gedi footprint ie. target!=0 """
            out = out.flatten()   # flatten(), reshape the array to 1 dimension 
            target = target.flatten()
            out = out[target !=0]  # remove the 0s from the GEDI data
            target = target[target != 0]
            loss = (abs(target - out)).mean()
            return(loss)
        return MAE
    elif name == "SIG":
        def sigLoss(out,target, alpha =10, lamda=0.85, epsilon =1e-8):
            """sigLoss , calculated only where there is a gedi footprint ie. target!=0 """
            out = out.flatten()   # flatten(), reshape the array to 1 dimension 
            target = target.flatten()
            out = out[target !=0]  # remove the 0s from the GEDI data
            target = target[target != 0]
            out = torch.clamp(out, min=epsilon)
            # target = torch.clamp(target, min=epsilon)
            valid_pix_num = target.numel()  # Or len(target)
            diff = torch.log(out) - torch.log(target)
            
            diff_squared_sum = torch.sum(diff ** 2)
            diff_sum_squared = torch.sum(diff) ** 2
            
            loss = alpha * torch.sqrt(diff_squared_sum / valid_pix_num - lamda * diff_sum_squared / (valid_pix_num ** 2))
            return(loss)
        return sigLoss
    elif name == "HUBER":
        def huber_loss(out, target):
            out = out.flatten()   # flatten(), reshape the array to 1 dimension 
            target = target.flatten()
            out = out[target !=0]  # remove the 0s from the GEDI data
            target = target[target != 0]

            delta=3.0  # need to adjust this delta based on the needs, a smaller delta make the loss function as MSE, 
            #            larger (>1) like MAE, normally between 0 and 1 is a good balance between MAE and MSE
            # above delta (m) is linear, below is non-linear
            error = out - target
            abs_error = torch.abs(error)
            quadratic_part = torch.clamp(abs_error, max=delta)
            linear_part = abs_error - quadratic_part
            loss = 0.5 * quadratic_part**2 + delta * linear_part
            return torch.mean(loss)
        return huber_loss

    elif  name == "CE":
        #def focal_loss(out, target, gamma=2.0, alpha=[0.25, 0.5, 0.35]):
        def focal_loss(out, target, gamma=2.0, alpha_per_class=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], alpha_true = True):
            # Masking
            out = out.permute(0, 2, 3, 1)  # Reshape and filter; shape becomes [num_valid_pixels, num_classes] (BxWxH)xC
            out = out.reshape(-1, out.shape[3])
            target = target.flatten()
            mask = (target != -1)
            out = out[mask]
            target = target[mask]

            import torch.nn.functional as F
    
            # Convert target to one-hot encoding
            num_classes = out.size(1)
            target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    
            # Calculate cross entropy terms
            log_prob = F.log_softmax(out, dim=1)
            cross_entropy = -torch.sum(target_one_hot * log_prob, dim=1)

            # Convert alpha_per_class to tensor
            alpha_per_class = torch.tensor(alpha_per_class, dtype=torch.float32)
    
            # Calculate focal loss
            if not alpha_true:
                alpha = torch.ones(num_classes, dtype=torch.float32).to(out.device)
            else:
                 alpha = alpha_per_class[target].to(out.device)
    
            # Calculate focal weights
            pt = torch.exp(-cross_entropy)
            focal_weight = (alpha * (1 - pt) ** gamma).unsqueeze(0)
    
            loss = torch.mean(focal_weight * cross_entropy)
    
            return loss
        return focal_loss
    else:
        raise ValueError(f"Unknown function: {name}")
