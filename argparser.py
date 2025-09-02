import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', default='screw_bag',
                        help='category')
    parser.add_argument('-o', '--output_dir', default='./results/')
    parser.add_argument('-w', '--weights', default='models/teacher_medium.pth')
    parser.add_argument('-i', '--imagenet_train_path', default='./data/imagenet/train',)
    parser.add_argument('--mvtec_loco_path', default='./data/mvtec_loco'),
    parser.add_argument('--mvtec_loco_seg_path', default='./data/mvtec_loco_composition_maps/',)
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()