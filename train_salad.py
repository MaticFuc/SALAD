#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools
import os
import random
from tqdm import tqdm
from effad_networks import get_autoencoder, get_pdn_small, get_pdn_medium
from salad_dataset import ImageFolderWithoutTarget, ImageFolderWithPath, ImageFolderWithoutTargetWithSeg, ImageFolderWithPathWithSeg, InfiniteDataloader
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from collections import defaultdict
from ae import AutoEncoder
from unet import UNet
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from logger import log
from dice_loss import DiceLoss
from argparser import get_argparse

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256


default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

no_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

normalize = transforms.Compose([
])


transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))


def main():
    config = get_argparse()
    
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    dataset_path = config.mvtec_loco_path
    seg_dataset_path = config.mvtec_loco_seg_path

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, config.category)
    os.makedirs(train_output_dir, exist_ok=True)

    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.category, 'train'),
        transform=transforms.Lambda(train_transform))
    full_train_seg_set = ImageFolderWithoutTarget(
            os.path.join(seg_dataset_path, config.category, 'train'),
            transform=transforms.Lambda(train_transform))
    full_train_set.seg = False
    full_train_seg_set.seg = True

    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.category, 'test'),
        transform=default_transform)
    test_seg_set = ImageFolderWithoutTarget(
        os.path.join(seg_dataset_path, config.category, 'test'),
        transform=default_transform)
    test_set.seg = False
    test_seg_set.seg = True

    train_set = ImageFolderWithoutTargetWithSeg(full_train_set, full_train_seg_set)
    test_set = ImageFolderWithPathWithSeg(test_set, test_seg_set)

    validation_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.category, 'validation'),
        transform=transforms.Lambda(train_transform))
    validation_seg_set = ImageFolderWithoutTarget(
        os.path.join(seg_dataset_path, config.category, 'validation'),
        transform=transforms.Lambda(train_transform))
    validation_set.seg = False
    validation_seg_set.seg = True
    validation_set = ImageFolderWithoutTargetWithSeg(validation_set, validation_seg_set)

    

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_set.seg = False
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=1)
    train_loader_infinite = InfiniteDataloader(train_loader)
    # create models
    
    teacher = get_pdn_medium(out_channels=out_channels)
    student = get_pdn_medium(out_channels=2 * out_channels)
    teacher = torch.load(config.weights)
    autoencoder = get_autoencoder(out_channels=out_channels)
    comp_ae = AutoEncoder({})
    comp_unet = UNet({})


    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()
    comp_ae.train()
    comp_unet.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        # comp_disc.cuda()
        comp_ae.cuda()
        comp_unet.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam([{"params": list(student.parameters()) + list(autoencoder.parameters())},
                                  {"params": list(comp_ae.parameters()) + list(comp_unet.parameters()), "lr":1e-5}],
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    
    weights = get_weights(train_loader)
    focal_loss = sigmoid_focal_loss
    multiclass_focal_loss = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=torch.tensor(weights),
        gamma=2,
        reduction='mean',
        force_reload=False
    ).cuda()
    dice_loss_f = DiceLoss(weights).cuda()
    
    
    train_set.train = True

    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (img, seg, anom_seg, mask, _), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        image_st, image_ae = img
        image_st = normalize(image_st)
        image_ae = normalize(image_ae)
        anom_seg = torch.cat([anom_seg, seg],dim=0)
        mask = torch.cat([mask,torch.zeros_like(mask)],dim=0)
        seg = torch.cat([seg, seg],dim=0)
        seg = seg.argmax(dim=1)
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            seg = seg.cuda()
            anom_seg = anom_seg.cuda()
            mask = mask.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output)**2
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)

        seg_recon = comp_ae(anom_seg).softmax(dim=1)
        unet_input = torch.cat([anom_seg, seg_recon], dim=1)

        pred_mask = comp_unet(unet_input).squeeze(1)


        loss_comp_recon = multiclass_focal_loss(seg_recon, seg) + dice_loss_f(seg_recon, seg)
        loss_comp_mask = 5*focal_loss(pred_mask, mask).mean() + F.l1_loss(nn.Sigmoid()(pred_mask), mask)

        loss_total = loss_st + loss_ae + loss_stae + loss_comp_recon + loss_comp_mask

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                # "Current loss: {:.4f}".format(loss_total.item()))
                "Current loss: {:.4f}, comp recon loss {:.4f}, comp disc loss {:.4f}".format(loss_total.item(), loss_comp_recon.item(), loss_comp_mask.item()))


        if iteration % 10000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                             'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                             'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                 'autoencoder_tmp.pth'))
            torch.save(comp_ae, os.path.join(train_output_dir,
                                                 'comp_autoencoder_tmp.pth'))
            torch.save(comp_unet, os.path.join(train_output_dir,
                                                 'comp_unet_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            teacher.eval()
            student.eval()
            autoencoder.eval()
            comp_ae.eval()
            comp_unet.eval()
            # comp_disc.eval()
            train_set.train = False


            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, desc='Intermediate map normalization')
            q_eff_start, q_eff_end, q_seg_start, q_seg_end = score_normalization(
                validation_loader=validation_loader, teacher=teacher, student=student, comp_ae=comp_ae, comp_unet=comp_unet,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end, desc='Final score normalization')
            feature_vectors_mean, feature_vectors_covinv, feature_vectors_mean_seg, feature_vectors_covinv_seg, feature_vectors_mean_seg_area, feature_vectors_covinv_seg_area = extract_features_mahalanobis(train_loader, train_set, student, teacher_mean, teacher_std)
            q_start_mah, q_end_mah = map_normalization_mahalanobis(validation_loader, student, teacher_mean, teacher_std, feature_vectors_covinv, feature_vectors_mean, feature_vectors_covinv_seg, feature_vectors_mean_seg, feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area)
            auc, auc_img, auc_mlp, auc_comp = test(
                test_set=test_set, teacher=teacher, student=student, comp_ae=comp_ae, comp_unet=comp_unet,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, feature_vectors_covinv=feature_vectors_covinv, feature_vectors_mean=feature_vectors_mean, feature_vectors_covinv_seg=feature_vectors_covinv_seg, feature_vectors_mean_seg=feature_vectors_mean_seg, feature_vectors_covinv_seg_area=feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area=feature_vectors_mean_seg_area,
                q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_eff_start=q_eff_start, q_eff_end=q_eff_end, q_ae_end=q_ae_end, q_seg_start=q_seg_start, q_seg_end=q_seg_end, q_start_mah=q_start_mah, q_end_mah=q_end_mah, desc='Intermediate inference')
            print('Intermediate image auc: {:.4f}, img {:.4f}, maha {:.4f}, comp {:.4f}'.format(auc, auc_img, auc_mlp, auc_comp))

            results = {
                "Iteration": [iteration],
                "Category": [config.category],
                "AUC": [auc],
                "AUC Img": [auc_img],
                "AUC Maha": [auc_mlp],
                "AUC Comp": [auc_comp]
            }
            log(train_output_dir,results)
            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()
            comp_ae.train()
            comp_unet.train()
            train_set.train = True

    train_set.train = False
    teacher.eval()
    student.eval()
    autoencoder.eval()
    comp_ae.eval()
    comp_unet.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))
    torch.save(comp_ae, os.path.join(train_output_dir,
                                                 'comp_autoencoder_final.pth'))
    torch.save(comp_unet, os.path.join(train_output_dir,
                                            'comp_unet_final.pth'))


    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    q_eff_start, q_eff_end, q_seg_start, q_seg_end = score_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student, comp_ae=comp_ae, comp_unet=comp_unet,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end, desc='Final score normalization')
    feature_vectors_mean, feature_vectors_covinv, feature_vectors_mean_seg, feature_vectors_covinv_seg, feature_vectors_mean_seg_area, feature_vectors_covinv_seg_area = extract_features_mahalanobis(train_loader, train_set, student, teacher_mean, teacher_std)
    q_start_mah, q_end_mah = map_normalization_mahalanobis(validation_loader, student, teacher_mean, teacher_std, feature_vectors_covinv, feature_vectors_mean, feature_vectors_covinv_seg, feature_vectors_mean_seg, feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area)
    auc, auc_img, auc_mlp, auc_comp = test(
        test_set=test_set, teacher=teacher, student=student, comp_ae=comp_ae, comp_unet=comp_unet,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, feature_vectors_covinv=feature_vectors_covinv, feature_vectors_mean=feature_vectors_mean, feature_vectors_covinv_seg=feature_vectors_covinv_seg, feature_vectors_mean_seg=feature_vectors_mean_seg, feature_vectors_covinv_seg_area=feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area=feature_vectors_mean_seg_area,
        q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_eff_start=q_eff_start, q_eff_end=q_eff_end, q_ae_end=q_ae_end, q_seg_start=q_seg_start, q_seg_end=q_seg_end, q_start_mah=q_start_mah, q_end_mah=q_end_mah, desc='Final inference')
    print('Final image auc: {:.4f}, img {:.4f}, maha {:.4f}, comp {:.4f}'.format(auc, auc_img, auc_mlp, auc_comp))
    results = {
        "Iteration": [iteration],
        "Category": [config.category],
        "AUC": [auc],
        "AUC Img": [auc_img],
        "AUC Maha": [auc_mlp],
        "AUC Comp": [auc_comp]
    }
    log(train_output_dir,results)

def test(test_set, teacher, student, autoencoder, comp_ae, comp_unet, 
         teacher_mean, teacher_std,
         feature_vectors_covinv, feature_vectors_mean, feature_vectors_covinv_seg, feature_vectors_mean_seg, feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area,
         q_st_start, q_st_end, q_ae_start, q_ae_end, q_eff_start, q_eff_end, q_seg_start, q_seg_end, q_start_mah, q_end_mah,
         desc='Running inference'):
    y_true = {"good":[],"logical_anomalies":[],"structural_anomalies":[]}
    y_score_no_mah = {"good":[],"logical_anomalies":[],"structural_anomalies":[]}
    y_score_mah = {"good":[],"logical_anomalies":[],"structural_anomalies":[]}
    y_score_comp = {"good":[],"logical_anomalies":[],"structural_anomalies":[]}
    y_score = {"good":[],"logical_anomalies":[],"structural_anomalies":[]}
    y_score_mah_all = []
    y_score_no_mah_all = []
    for image, seg, target, path in tqdm(test_set, desc=desc):

        image = image.unsqueeze(0)
        if on_gpu:
            image = image.cuda()
            seg = seg.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        
        map_combined = (map_combined - q_eff_start) / q_eff_end
        map_combined = map_combined[0, 0].cpu().numpy()
        

        map_comp = predict_comp_map(seg, comp_ae, comp_unet).unsqueeze(0)
        map_comp = (map_comp - q_seg_start) / q_seg_end
        map_comp = map_comp[0, 0].cpu().numpy()


        mahalanobis_score = predict_mahalanobis(image=image, seg=seg, teacher=student, teacher_mean=teacher_mean, feature_vectors_covinv=feature_vectors_covinv, feature_vectors_mean=feature_vectors_mean, feature_vectors_covinv_seg=feature_vectors_covinv_seg, feature_vectors_mean_seg=feature_vectors_mean_seg, feature_vectors_covinv_seg_area=feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area=feature_vectors_mean_seg_area,
            teacher_std=teacher_std, q_start=q_start_mah, q_end=q_end_mah)
        defect_class = os.path.basename(os.path.dirname(path))

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined) + mahalanobis_score + np.max(map_comp)
        y_score_img_no_mlp = np.max(map_combined)
        y_true[defect_class].append(y_true_image)
        y_score_mah[defect_class].append(mahalanobis_score)
        y_score_no_mah[defect_class].append(y_score_img_no_mlp)
        y_score_comp[defect_class].append(np.max(map_comp))
        y_score[defect_class].append(y_score_image)

        y_score_mah_all.append(mahalanobis_score)
        y_score_no_mah_all.append(y_score_img_no_mlp)
    
    auc_log = roc_auc_score(y_true=y_true['good']+y_true['logical_anomalies'], y_score=y_score['good'] + y_score['logical_anomalies'])
    auc_str = roc_auc_score(y_true=y_true['good']+y_true['structural_anomalies'], y_score=y_score['good']+y_score['structural_anomalies'])
    auc = 0.5*(auc_log+auc_str)
    auc_log_mah = roc_auc_score(y_true=y_true['good']+y_true['logical_anomalies'], y_score=y_score_mah['good'] + y_score_mah['logical_anomalies'])
    auc_str_mah = roc_auc_score(y_true=y_true['good']+y_true['structural_anomalies'], y_score=y_score_mah['good']+y_score_mah['structural_anomalies'])
    auc_mlp = 0.5*(auc_log_mah+auc_str_mah)
    auc_log_no_mah = roc_auc_score(y_true=y_true['good']+y_true['logical_anomalies'], y_score=y_score_no_mah['good'] + y_score_no_mah['logical_anomalies'])
    auc_str_no_mah = roc_auc_score(y_true=y_true['good']+y_true['structural_anomalies'], y_score=y_score_no_mah['good']+y_score_no_mah['structural_anomalies'])
    auc_img = 0.5*(auc_log_no_mah+auc_str_no_mah)
    auc_log_comp = roc_auc_score(y_true=y_true['good']+y_true['logical_anomalies'], y_score=y_score_comp['good'] + y_score_comp['logical_anomalies'])
    auc_str_comp = roc_auc_score(y_true=y_true['good']+y_true['structural_anomalies'], y_score=y_score_comp['good']+y_score_comp['structural_anomalies'])
    auc_comp = 0.5*(auc_log_comp+auc_str_comp)

    print("AUC Scores for different parts")
    print(f"All Logical: {auc_log*100}, Struct: {auc_str*100}")
    print(f"Maha Logical: {auc_log_mah*100}, Struct: {auc_str_mah*100}")
    print(f"Img Logical: {auc_log_no_mah*100}, Struct: {auc_str_no_mah*100}")
    print(f"Comp Logical: {auc_log_comp*100}, Struct: {auc_str_comp*100}")
    print("")
    return auc * 100, auc_img * 100 , auc_mlp * 100, auc_comp*100

@torch.no_grad()
def predict_comp_map(seg, comp_ae, comp_unet):
    seg = seg.unsqueeze(0)
    seg_recon = comp_ae(seg).softmax(dim=1)
    unet_input = torch.cat([seg, seg_recon], dim=1)

    pred_mask = comp_unet(unet_input)
    pred_mask = nn.Sigmoid()(pred_mask)
    pred_mask = torch.nn.functional.interpolate(
            pred_mask, (256, 256), mode='bilinear').squeeze(1)
    return pred_mask

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    map_st = torch.nn.functional.interpolate(
            map_st, (256, 256), mode='bilinear')
    map_ae = torch.nn.functional.interpolate(
            map_ae, (256, 256), mode='bilinear')
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def predict_mahalanobis(image, seg, teacher, teacher_mean, teacher_std, feature_vectors_covinv, feature_vectors_mean, feature_vectors_covinv_seg, feature_vectors_mean_seg, feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area, q_start=None, q_end=None):
    teacher_output = teacher(image)[:,:384,:,:]
    
    teacher_avg_vec = teacher_output.mean(dim=(2,3)).squeeze().detach().cpu().numpy() -  feature_vectors_mean
    mahalanobis_distance = {}
    mahalanobis_distance["full"] = np.sqrt(
        max(
            0,
            np.dot(
                np.dot(teacher_avg_vec, feature_vectors_covinv),
                teacher_avg_vec,
            ),
        )
    )

    for k in feature_vectors_covinv_seg.keys():
        teacher_avg_vec, area = get_mahalanobis_prediction(seg.unsqueeze(0), k, teacher_output)
        teacher_avg_vec = teacher_avg_vec - feature_vectors_mean_seg[k]
        mahalanobis_distance[k] = np.sqrt(
            max(
                0,
                np.dot(
                    np.dot(teacher_avg_vec, feature_vectors_covinv_seg[k]),
                    teacher_avg_vec,
                ),
            )
        )
    
    if q_start is not None:
        full_score = 0
        for k in q_start.keys():

            if q_end[k] != 0:
                full_score_k = (mahalanobis_distance[k] - q_start[k]) / (q_end[k])
            else:
                full_score_k = 0
            if k == "full":
                full_score += full_score_k
            else:
                full_score += full_score_k / (len(q_start.keys()) - 1)
        return full_score
    else:    
        return mahalanobis_distance

@torch.no_grad()
def score_normalization(validation_loader, teacher, student, autoencoder, comp_ae, comp_unet,
                      teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end, desc='Score normalization'):
    eff_scores = []
    comp_scores = []
    # ignore augmented ae image
    with torch.no_grad():
        for image, seg, _, _, _ in tqdm(validation_loader, desc=desc):
            image = image[0]
            seg = seg[0]
            if on_gpu:
                image = image.cuda()
                seg = seg.cuda()
            image = normalize(image)
            map_combined, map_st, map_ae = predict(
                image=image, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end)
            
            eff_score = torch.max(map_combined).cpu().numpy()
            
            map_comp = predict_comp_map(seg, comp_ae, comp_unet).max().cpu().numpy()
            
            eff_scores.append(eff_score)
            comp_scores.append(map_comp)


    q_eff_start = np.mean(eff_scores)
    q_eff_end = np.std(eff_scores)
    q_seg_start = np.mean(comp_scores)
    q_seg_end = np.std(comp_scores)

    return q_eff_start, q_eff_end, q_seg_start, q_seg_end


@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, seg, _, _, _ in tqdm(validation_loader, desc=desc):
        image = image[0]
        
        seg = seg[0]
        if on_gpu:
            image = image.cuda()
            seg = seg.cuda()
        image = normalize(image)
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        
        
        maps_st.append(map_st)
        maps_ae.append(map_ae)


    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)

    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)


    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    
    for train_image, _, _, _, _ in tqdm(train_loader, desc='Computing teacher mean and std'):
        train_image = train_image[0]
        if on_gpu:
            train_image = train_image.cuda()
        train_image = normalize(train_image)
        teacher_output = teacher(train_image)
        mean_outputs.append(teacher_output)
    mean_outputs = torch.cat(mean_outputs,dim=0)
    channel_mean = mean_outputs.mean(dim=(0,2,3))
    channel_std = mean_outputs.std(dim=(0,2,3))
    channel_mean = channel_mean[None, :, None, None]
    channel_std = channel_std[None, :, None, None]
    
    
    return channel_mean, channel_std

def get_mahalanobis_prediction(seg, k, teacher_output):
    mask = seg[:,k,:,:].unsqueeze(0)
    area_hr = mask.mean().detach().cpu().numpy()
    mask = transforms.Resize((56,56))(mask)
    teacher_avg_vec_mask = mask * teacher_output
    area = mask.sum().detach().cpu().numpy()
    teacher_avg_vec_mask = teacher_avg_vec_mask.sum(dim=(0,2,3)).squeeze().detach().cpu().numpy()
    if area != 0:
        teacher_avg_vec_mask = teacher_avg_vec_mask / area
    area = np.array([area_hr])
    return teacher_avg_vec_mask, area

@torch.no_grad()
def extract_features_mahalanobis(train_loader, train_set, teacher, teacher_mean, teacher_std):
    tqdm_obj = tqdm(range(len(train_set)), desc="Mahalanobis Feature Extractor")
    feat_vectors = []
    feat_vectors_seg = defaultdict(list)
    feat_vectors_seg_area = defaultdict(list)
    for iteration, (img, seg, _, _, _) in zip(
            tqdm_obj, train_loader):
        image_st, image_ae = img
        
        if on_gpu:
            image_st = image_st.cuda()
            seg = seg.cuda()

        image_st = normalize(image_st)
        with torch.no_grad():
            teacher_output_st = teacher(image_st)[:,:384,:,:]
    
            teacher_avg_vec = teacher_output_st.mean(dim=(2,3)).squeeze().detach().cpu().numpy()
            feat_vectors.append(teacher_avg_vec)
            
            for k in range(seg.shape[1]):
                teacher_avg_vec_mask, area = get_mahalanobis_prediction(seg, k, teacher_output_st)
                feat_vectors_seg[k].append(teacher_avg_vec_mask)
                feat_vectors_seg_area[k].append(area)
    feat_vectors = np.array(feat_vectors)
    
    feature_vectors_mean = np.mean(feat_vectors, axis=0)
    cov = LedoitWolf().fit(feat_vectors).covariance_
    feature_vectors_covinv = np.linalg.pinv(cov)

    feature_vectors_mean_seg = defaultdict(int)
    feature_vectors_covinv_seg = defaultdict(int)
    feature_vectors_mean_seg_area = defaultdict(int)
    feature_vectors_covinv_seg_area = defaultdict(int)

    for k in feat_vectors_seg.keys():
        feat_vectors_seg_k = np.array(feat_vectors_seg[k])
        feature_vectors_mean_seg_k = np.mean(feat_vectors_seg_k, axis=0)
        cov = LedoitWolf().fit(feat_vectors_seg_k).covariance_
        feature_vectors_covinv_seg_k = np.linalg.pinv(cov)
        feature_vectors_mean_seg[k] = feature_vectors_mean_seg_k
        feature_vectors_covinv_seg[k] = feature_vectors_covinv_seg_k

        feat_vectors_seg_area_k = np.array(feat_vectors_seg_area[k])
        feature_vectors_mean_seg_area_k = np.mean(feat_vectors_seg_area_k, axis=0)
        cov = LedoitWolf().fit(feat_vectors_seg_area_k).covariance_
        feature_vectors_covinv_seg_area_k = np.linalg.pinv(cov)
        feature_vectors_mean_seg_area[k] = feature_vectors_mean_seg_area_k
        feature_vectors_covinv_seg_area[k] = feature_vectors_covinv_seg_area_k
    return feature_vectors_mean, feature_vectors_covinv, feature_vectors_mean_seg, feature_vectors_covinv_seg, feature_vectors_mean_seg_area, feature_vectors_covinv_seg_area


@torch.no_grad()
def map_normalization_mahalanobis(validation_loader, teacher,
                      teacher_mean, teacher_std, feature_vectors_covinv, feature_vectors_mean, feature_vectors_covinv_seg, feature_vectors_mean_seg, feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area, desc='Map normalization Mahalanobis'):
    maps = defaultdict(list)

    for image, seg, _, _, _ in tqdm(validation_loader, desc=desc):
        image = image[0]
        
        if on_gpu:
            image = image.cuda()
            seg = seg.cuda()[0,:]
        image = normalize(image)
        scores = predict_mahalanobis(
            image=image, seg=seg, teacher=teacher, teacher_mean=teacher_mean, feature_vectors_covinv=feature_vectors_covinv, feature_vectors_mean=feature_vectors_mean, feature_vectors_covinv_seg=feature_vectors_covinv_seg, feature_vectors_mean_seg=feature_vectors_mean_seg, feature_vectors_covinv_seg_area=feature_vectors_covinv_seg_area, feature_vectors_mean_seg_area=feature_vectors_mean_seg_area,
            teacher_std=teacher_std)
        

        for k in scores.keys():
            maps[k].append(scores[k])
    q_start = {}
    q_end = {}
    for k in maps.keys():
        maps_k = maps[k]
        q_start[k] = np.mean(maps_k)
        q_end[k] = np.std(maps_k)#torch.quantile(maps_k, q=0.995)

    return q_start, q_end

@torch.no_grad
def get_weights(train_loader):
    counts = defaultdict(int)
    for _, seg, _, _, _ in tqdm(train_loader, desc="CE Weight Calculation"):
        seg_mask = seg
        for j in range(seg.shape[1]):
            cnt = seg_mask == j
            cnt = cnt.int()
            cnt = cnt.sum()
            counts[j] += cnt
    total_sum = sum([counts[i] for i in counts.keys()])
    weights = list([total_sum / counts[i] if counts[i] != 0 else 1 for i in counts.keys()])
    return torch.FloatTensor(weights)

if __name__ == '__main__':
    main()
