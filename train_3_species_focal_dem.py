import os
from tqdm import tqdm
import datetime
import pytz
from model import UNet
import torch
from utils3_species_focal_dem import get_dataloader, get_loss_function
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, confusion_matrix
import yaml 
import logging
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

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
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch Size during training') #from 64
    parser.add_argument('--epoch', default=500, type=int, help='epoch to run')
    parser.add_argument('--num_worker', default=0, type=int, help='number of workers to load data')
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
    parser.add_argument('--load_model', default=False, help='use pretrained model')
    parser.add_argument('--slope', default=True, help='use slope as input')
    parser.add_argument('--height', default=True, help='use height as input')
    parser.add_argument('--aspect', default=True, help='use aspect as input')
    parser.add_argument("--loss_function", type=str, choices=["MSE", "MAE", "SIG", "HUBER","CE"], 
                    required=True, help="Specify the loss function to use")
    parser.add_argument('--ckpts', default=ckpts_path, help='ckpts')
    parser.add_argument('--gpu', type=str, default='0,1', help='specify GPU devices')
    parser.add_argument('--model_time', type=str, default=MODEL_TIME, help='model training start time')
    parser.add_argument('--exp', type=str, default=experiment_name, help='experiment name')
    return parser.parse_args()

def main(args,LOGGER):
    LOGGER.info(f'Experiment name: {args.exp} \nModel train start at: {args.model_time}\n\n{args}')

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
    """
    # Augmentations 정의
    horizontal_flip = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])  # 가로로 뒤집기
    vertical_flip = transforms.Compose([transforms.RandomVerticalFlip(p=1.0)])      # 세로로 뒤집기
    rotate_90 = transforms.Compose([transforms.RandomRotation(degrees=(90, 90))])   # 90도 회전
    rotate_180 = transforms.Compose([transforms.RandomRotation(degrees=(180, 180))]) # 180도 회전
    rotate_270 = transforms.Compose([transforms.RandomRotation(degrees=(270, 270))]) # 270도 회전
    """
    # 원본 데이터셋
    train_loader = get_dataloader(split='Train', **DATALOADER_PARAMS)#.dataset
    """
    # 각각의 augmentation이 적용된 데이터셋 생성
    horizontal_flip_dataset = get_dataloader(split='Train', transform=horizontal_flip, **DATALOADER_PARAMS).dataset
    vertical_flip_dataset = get_dataloader(split='Train', transform=vertical_flip, **DATALOADER_PARAMS).dataset
    rotate_90_dataset = get_dataloader(split='Train', transform=rotate_90, **DATALOADER_PARAMS).dataset
    rotate_180_dataset = get_dataloader(split='Train', transform=rotate_180, **DATALOADER_PARAMS).dataset
    rotate_270_dataset = get_dataloader(split='Train', transform=rotate_270, **DATALOADER_PARAMS).dataset

    # 원본 데이터셋과 모든 augmentation 데이터셋을 합침
    combined_dataset = ConcatDataset([original_dataset, 
                                      horizontal_flip_dataset, 
                                    vertical_flip_dataset, 
                                    rotate_90_dataset, 
                                    rotate_180_dataset, 
                                    rotate_270_dataset])

    # Dataloader로 변환 (DATALOADER_PARAMS는 그대로 유지)
    train_loader = get_dataloader(split='Train', **DATALOADER_PARAMS)
    """
    val_loader = get_dataloader(split = 'Validation', **DATALOADER_PARAMS) 
    test_loader = get_dataloader(split = 'Test', **DATALOADER_PARAMS) 
    LOGGER.info(f'\ntrain dataset: {len(train_loader.dataset)} \nvalidation dataset: {len(val_loader.dataset)} \ntest dataset: {len(test_loader.dataset)}')

    # Model
    #if args.slope:
    #    n_channels =len(DATALOADER_PARAMS.get('bands_s2'))#+len(DATALOADER_PARAMS.get('bands_s1'))+len(DATALOADER_PARAMS.get('bands_slope'))
    #else:
    #    n_channels =len(DATALOADER_PARAMS.get('bands_s2'))#+len(DATALOADER_PARAMS.get('bands_s1'))
    n_channels =len(DATALOADER_PARAMS.get('bands_s2'))+len(DATALOADER_PARAMS.get('bands_slope'))+len(DATALOADER_PARAMS.get('bands_aspect'))+len(DATALOADER_PARAMS.get('bands_height'))

    # n_classes = 8
    n_classes = 9

    model = UNet(n_channels=n_channels, n_classes=n_classes)
    model = model.cuda()
    loss_function_ = get_loss_function(args.loss_function)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True,patience = 10, factor = 0.9)

    def calculate_metrics(preds, labels, num_classes):
        preds_forcal = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        labels_forcal = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        # Calculate TP, FP, FN, Precision, Recall, and F1 Score for each class
        tp = np.zeros(num_classes)
        fp = np.zeros(num_classes)
        fn = np.zeros(num_classes)

        for cls in range(num_classes):
            tp[cls] = np.sum((preds_forcal == cls) & (labels_forcal == cls))
            fp[cls] = np.sum((preds_forcal == cls) & (labels_forcal != cls))
            fn[cls] = np.sum((preds_forcal != cls) & (labels_forcal == cls))

        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        accuracy = np.zeros(num_classes)

        # Calculate micro-level metrics
        micro_tp = np.sum(tp)
        micro_fp = np.sum(fp)
        micro_fn = np.sum(fn)

        if micro_tp + micro_fp > 0:
            micro_precision = micro_tp / (micro_tp + micro_fp)
        else:
            micro_precision = 0.0

        if micro_tp + micro_fn > 0:
            micro_recall = micro_tp / (micro_tp + micro_fn)
        else:
            micro_recall = 0.0

        if micro_precision + micro_recall > 0:
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        else:
            micro_f1 = 0.0

        return micro_f1, micro_precision, micro_recall

    torch.cuda.empty_cache() # Reduce model size
    if args.loss_function == "CE":
        best_val_metric = -0.00001
    else:
        best_val_metric = 1000

    if args.load_model:
        checkpoint_path=args.ckpts
        checkpoint = torch.load(checkpoint_path)
        # Load the model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        new_lr = 0.000430
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        if args.loss_function == "CE":
            best_val_metric = checkpoint['val_f1']
        else:
            best_val_metric = checkpoint['val_mae']
        epoch_saved = checkpoint['epoch']
        saved_epoch = 0
        LOGGER.info(f'Loaded model checkpoint from: {checkpoint_path} \nBest validation metric: {best_val_metric}\nsaved at epoch {epoch_saved}')

    for epoch in range(args.epoch):
        #train_running_loss = 0.0 # metric for Tensorboard
        #train_running_RMSE = 0.0 # metric for Tensorboard
        #train_running_MAE = 0.0 # metric for Tensorboard
        out_graph = np.array([]) # For visu on Tensorboard
        labels_graph = np.array([]) # For visu on Tensorboard
        LOGGER.info(f"------------current epoch : {epoch+1}------------")
       
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        # Training
        loop = tqdm(train_loader)
        loss_batch = []
        outs = []
        num_batches = len(loop)
        print("Number of batches per epoch:", num_batches)
        labels_list = []
        train_f1_list = []
        train_precision_list = []
        train_recall_list = []
        for cls in range(n_classes):
            globals()[f'batch_train_f1_{cls+1}_list'] = []
            globals()[f'batch_train_precision_{cls+1}_list'] = []
            globals()[f'batch_train_recall_{cls+1}_list'] = []

        correct = 0
        total = 0

        for batch_idx,(images,labels,bin_labels) in enumerate(loop):
            images = images.cuda().float()
            images = torch.nan_to_num(images) # to remove nans in S1 images
            labels = labels.cuda().long() 

            # make the label start from 0 to not make error at loss.backward
            labels -= 1
            pft_masked = torch.where((labels == 0) | (labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5) | (labels == 6)  | (labels == 7) | (labels == 8), labels, torch.tensor(-1).cuda())            
            
            out = model(images)  # where the model applied
            # print(out.shape)
            #labels = torch.nan_to_num(labels)
            pft_masked = torch.nan_to_num(pft_masked)
            if args.loss_function == "CE":
                loss = loss_function_(out, pft_masked.long())
            else:
                loss = loss_function_(out.float(), labels.float())
                
                        
            optimizer.zero_grad() # where the model is trained
            loss.backward() # where the model is trained
            optimizer.step() # where the model is trained, search on google
            loss_batch.append(loss.detach().item())
            
            # Cal MAE or accuracy based on loss function
            if args.loss_function == "CE":
                probs = F.softmax(out, dim=1) 
                conf, preds = torch.max(probs, 1)
                below_threshold_mask = (conf < 0.5)  # Mask for confidence < 50%
                preds[below_threshold_mask] = 9  # Predict 10 when confidence is below 50% for labels 0~2

                

                # Masking non-zero PFT pixels
                non_zero_mask = pft_masked != -1
                
                for cls_idx in range(n_classes):
                    class_mask = pft_masked == cls_idx
                    train_f1, train_precision, train_recall = calculate_metrics(preds[class_mask], (pft_masked[class_mask]), n_classes)
    
                    globals()[f'batch_train_f1_{cls_idx + 1}_list'].append(train_f1)
                    globals()[f'batch_train_precision_{cls_idx + 1}_list'].append(train_precision)
                    globals()[f'batch_train_recall_{cls_idx + 1}_list'].append(train_recall)
                
                # bring back the label range
                preds = preds + 1
                
                # Count correct predictions for all pixels
                #correct += torch.sum(preds == (labels + 1)).item()
                #total += labels.numel() 
                
                #corrects = torch.sum(preds == labels.data)
                #total += labels.size(0)*256*256 #256*256 image
                #correct += corrects.item()
                outs.append(preds.cpu().numpy())
                #labels_list.append((labels+1).cpu().numpy())
                batch_train_f1, batch_train_precision, batch_train_recall = calculate_metrics(preds[non_zero_mask], (pft_masked[non_zero_mask]+1), n_classes)
                train_f1_list.append(batch_train_f1)
                train_precision_list.append(batch_train_precision)
                train_recall_list.append(batch_train_recall)
                del out, loss, images, labels, pft_masked, batch_train_precision, batch_train_recall

            else:
                out = out.cpu().detach().numpy()
                labels = np.expand_dims(labels.cpu(), 1)
                            
                out = out[labels!=0]
                label = labels[labels!=0]
                outs.append(out)
                labels_list.append(label)
                del out, label, loss, images, labels
        

        if args.loss_function == "CE":
            train_f1 = np.mean(train_f1_list)
            train_precision = np.mean(train_precision_list)
            train_recall = np.mean(train_recall_list)

            for cls in range(n_classes):
                cls_train_f1 = np.mean(globals()[f'batch_train_f1_{cls+1}_list'])
                cls_train_precision = np.mean(globals()[f'batch_train_precision_{cls+1}_list'])
                cls_train_recall = np.mean(globals()[f'batch_train_recall_{cls+1}_list'])
                LOGGER.info(f'Train Label {cls+1} - F1: {cls_train_f1:.4f} || Precision: {cls_train_precision:.4f} || Recall: {cls_train_recall:.4f}')
            LOGGER.info(f'Train Loss: {np.mean(loss_batch):.4f} || F1 Score: {train_f1:.4f} || Train Precision: {train_precision:.4f} || Train Recall: {train_recall:.4f} \nlr: {optimizer.param_groups[0]["lr"]:.6f}\n')

        else:
            outs = np.concatenate(outs)
            labels_list = np.concatenate(labels_list)
            train_mae = mean_absolute_error(labels_list, outs)
            LOGGER.info(f'Train Loss: {np.mean(loss_batch):.4f} || Train MAE: {train_mae:.4f} \nlr: {optimizer.param_groups[0]["lr"]:.6f}\n')
        
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        torch.cuda.empty_cache()

        # =========== Evaluation part ===========
        with torch.no_grad():
            model.eval()
            # for visualization
            out_graph = np.array([])
            labels_graph = np.array([])
            
            outs = []
            labels_list = []
            val_f1_list = []
            val_precision_list = []
            val_recall_list = []
            for cls in range(n_classes):
                globals()[f'batch_val_f1_{cls+1}_list'] = []
                globals()[f'batch_val_precision_{cls+1}_list'] = []
                globals()[f'batch_val_recall_{cls+1}_list'] = []    
            
            val_loss_list = [] 
            val_correct = 0
            total_samples = 0

            loop = tqdm(val_loader)
            for batch_idx, (images,labels,bin_labels) in enumerate(loop):
                images = images.cuda().float()
                images = torch.nan_to_num(images) # to remove nans
                labels = labels.type(torch.LongTensor).cuda()

                # bring the labels to start from 0
                labels = labels - 1
                pft_masked = torch.where((labels == 0) | (labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5) | (labels == 6)  | (labels == 7) | (labels == 8), labels, torch.tensor(-1).cuda())    

                out = model(images)
                labels = torch.nan_to_num(labels)
                pft_masked = torch.nan_to_num(pft_masked)
                
                if args.loss_function == "CE":
                    # Calculate validation loss
                    val_loss_batch = loss_function_(out, pft_masked.long())

                    val_loss_list.append(val_loss_batch.detach().item())

                    probs = F.softmax(out, dim=1) 
                    conf, val_preds = torch.max(probs, 1)
                    below_threshold_mask = (conf < 0.5) # Mask for confidence < 50%
                    val_preds[below_threshold_mask] = 9  # Predict 10 when confidence is below 50% for labels 0~2


                    #outs.extend(preds.cpu().numpy().flatten())
                    #labels_list.extend(labels.cpu().numpy().flatten())

                    non_zero_mask = pft_masked != -1  # Masking non-zero PFT pixels

                    for cls_idx in range(n_classes):
                        class_mask = pft_masked == cls_idx
                        batch_val_f1, batch_val_precision, batch_val_recall = calculate_metrics(val_preds[class_mask], (pft_masked[class_mask]), n_classes)

                        globals()[f'batch_val_f1_{cls_idx + 1}_list'].append(batch_val_f1)
                        globals()[f'batch_val_precision_{cls_idx + 1}_list'].append(batch_val_precision)
                        globals()[f'batch_val_recall_{cls_idx + 1}_list'].append(batch_val_recall)
                    
                    # bring back the labels
                    val_preds = val_preds + 1

                    batch_val_f1, batch_val_precision, batch_val_recall = calculate_metrics(val_preds[non_zero_mask], (pft_masked[non_zero_mask]+1), n_classes)
                    val_f1_list.append(batch_val_f1)
                    val_precision_list.append(batch_val_precision)
                    val_recall_list.append(batch_val_recall)

                

                    # Count correct predictions for all pixels
                    #val_correct += torch.sum(preds == (labels + 1)).item()
                    #total_samples += labels.numel()

                    #val_correct += torch.sum(preds == labels).item()
                    #total_samples += labels.size(0)*256*256

                    del images, labels, out, pft_masked
                else:
                    labels = np.expand_dims(labels.cpu(), 1)
                    out = out.cpu().detach().numpy()
                    out = out[labels != 0]
                    label = labels[labels != 0]
                    outs.extend(out)
                    labels_list.extend(label)
                    del images, labels, out, label
            if args.loss_function == "CE":
                val_loss = np.mean(val_loss_list)
                val_f1 = np.mean(val_f1_list)
                val_precision = np.mean(val_precision_list)
                val_recall = np.mean(val_recall_list)

                LOGGER.info(f'Validation Loss: {val_loss:.4f} || F1 Score: {val_f1:.4f} || Validation Precision: {val_precision:.4f} || Validation Recall: {val_recall:.4f} \n')

                for cls in range(n_classes):
                    cls_val_f1 = np.mean(globals()[f'batch_val_f1_{cls+1}_list'])
                    cls_val_precision = np.mean(globals()[f'batch_val_precision_{cls+1}_list'])
                    cls_val_recall = np.mean(globals()[f'batch_val_recall_{cls+1}_list'])

                    LOGGER.info(f'Val Label {cls+1} - F1: {cls_val_f1:.4f} || Precision: {cls_val_precision:.4f} || Recall: {cls_val_recall:.4f}')

                if val_f1 > best_val_metric:
                    best_val_metric = val_f1
                    saved_epoch = epoch + 1
                    state = {
                    'epoch': saved_epoch,
                    'train_mae': train_mae if args.loss_function != "CE" else None,
                    'val_mae': val_mae if args.loss_function != "CE" else None,
                    'val_precision': val_precision if args.loss_function == "CE" else None,
                    'val_recall': val_precision if args.loss_function == "CE" else None,
                    'val_f1': val_f1 if args.loss_function == "CE" else None,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                    torch.save(state, os.path.join(args.model_path, f'best_model_{args.model_time}.pth'))
                LOGGER.info(f'model saved epoch: {saved_epoch}, validation metric: {best_val_metric:0.4f}')

            else:
                outs = np.concatenate(outs)
                labels_list = np.concatenate(labels_list)
                val_mae = mean_absolute_error(labels_list, outs)
                LOGGER.info(f'Validation Loss: {val_mae:.4f}')
                if val_mae < best_val_metric:
                    best_val_metric = val_mae
                    saved_epoch = epoch + 1
            """
            # Save the model if it improves
            if best_val_metric is not None:
                state = {
                    'epoch': saved_epoch,
                    'train_mae': train_mae if args.loss_function != "CE" else None,
                    'val_mae': val_mae if args.loss_function != "CE" else None,
                    'val_precision': val_precision if args.loss_function == "CE" else None,
                    'val_recall': val_precision if args.loss_function == "CE" else None,
                    'val_f1': val_f1 if args.loss_function == "CE" else None,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(args.model_path, f'best_model_{args.model_time}.pth'))
                LOGGER.info(f'model saved epoch: {saved_epoch}, validation metric: {best_val_metric:0.4f}')
            """
        torch.cuda.empty_cache()
        model.train()
        
if __name__ == "__main__":
    args = arg_parser()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"

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
    print("=================================")
    # makelog
    LOGGER = logging.getLogger(args.exp)

    # 로그의 출력 기준 설정
    LOGGER.setLevel(logging.INFO)

    #log 출력 형식
    formatter = logging.Formatter("%(message)s")

    # log를 console에 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    #log를 파일에 출력
    file_handler = logging.FileHandler(os.path.join(args.root_path,f"{args.model_time}.log"))
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    main(args,LOGGER)