from argparser import get_argparse
import os
import torch
from salad_dataset import ImageFolderWithoutTarget, ImageFolderWithPath, ImageFolderWithoutTargetWithSeg, ImageFolderWithPathWithSeg
from train_salad import teacher_normalization, map_normalization, score_normalization, extract_features_mahalanobis, test, map_normalization_mahalanobis, train_transform, default_transform
from logger import log
from torchvision import transforms
from torch.utils.data import DataLoader

seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

def test_all():

    config = get_argparse()
    seed = config.seed

    dataset_path = config.mvtec_loco_path
    seg_dataset_path = config.mvtec_loco_seg_path


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
    
    full_train_set = ImageFolderWithoutTargetWithSeg(full_train_set, full_train_seg_set)
    test_set = ImageFolderWithPathWithSeg(test_set, test_seg_set)


    train_set = full_train_set
    validation_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.category, 'validation'),
        transform=transforms.Lambda(train_transform))
    validation_seg_set = ImageFolderWithoutTarget(
        os.path.join(seg_dataset_path, config.category, 'validation'),
        transform=transforms.Lambda(train_transform))
    validation_set.seg = False
    validation_seg_set.seg = True
    validation_set = ImageFolderWithoutTargetWithSeg(validation_set, validation_seg_set)


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=1)
    

    teacher = torch.load(f"{train_output_dir}/teacher_final.pth")
    autoencoder = torch.load(f"{train_output_dir}/autoencoder_final.pth")
    student = torch.load(f"{train_output_dir}/student_final.pth")
    comp_ae = torch.load(f"{train_output_dir}/comp_autoencoder_final.pth")
    comp_unet = torch.load(f"{train_output_dir}/comp_unet_final.pth")
    

    # teacher frozen

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        comp_ae.cuda()
        comp_unet.cuda()

    teacher.eval()
    student.eval()
    autoencoder.eval()
    comp_ae.eval()
    comp_unet.eval()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)    

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
    print('Final image auc: {:.4f}, img {:.4f}, mlp {:.4f}, comp {:.4f}'.format(auc, auc_img, auc_mlp, auc_comp))
    results = {
        "Iteration": [-1],
        "Category": [config.category],
        "AUC": [auc],
        "AUC Img": [auc_img],
        "AUC Maha": [auc_mlp],
        "AUC Comp": [auc_comp]
    }
    log(train_output_dir,results)


if __name__ == '__main__':
    test_all()