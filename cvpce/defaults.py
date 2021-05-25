from . import utils

DATA_DIR = ('..', 'data')

COCO_IMG_DIR = utils.rel_path(*DATA_DIR, 'coco', 'val2017')
COCO_ANNOTATION_FILE = utils.rel_path(*DATA_DIR, 'coco', 'annotations', 'instances_val2017.json')

SKU110K_IMG_DIR = utils.rel_path(*DATA_DIR, 'SKU110K_fixed', 'images')
SKU110K_ANNOTATION_FILE = utils.rel_path(*DATA_DIR, 'SKU110K_fixed', 'annotations', 'annotations_val.csv')
SKU110K_SKIP = [
    'test_274.jpg', 'train_882.jpg', 'train_924.jpg', 'train_4222.jpg', 'train_5822.jpg', # corrupted images, won't load, TODO: re-export test for fairness
    'train_789.jpg', 'train_5007.jpg', 'train_6090.jpg', 'train_7576.jpg', # corrupted images, will load
    'train_104.jpg', 'train_890.jpg', 'train_1296.jpg', 'train_3029.jpg', 'train_3530.jpg', 'train_3622.jpg', 'train_4899.jpg', 'train_6216.jpg', 'train_7880.jpg', # missing most ground truth boxes
    'train_701.jpg', 'train_6566.jpg', # very poor images
]


GP_ROOT = (*DATA_DIR, 'Grocery_products')
GP_TRAIN_FOLDERS = (utils.rel_path(*GP_ROOT, 'Training'),)
GP_TEST_DIR = utils.rel_path(*GP_ROOT, 'Testing')
GP_ANN_DIR = utils.rel_path(*DATA_DIR, 'Planogram_Dataset', 'annotations')
GP_BASELINE_ANN_FILE = utils.rel_path(*DATA_DIR, 'Baseline', 'Grocery_products_coco_gt_object.csv')
GP_PLANO_DIR = utils.rel_path(*DATA_DIR, 'Planogram_Dataset', 'planograms')
GP_TEST_VALIDATION_SET = ['s1_15.csv', 's2_3.csv', 's2_30.csv', 's2_143.csv', 's2_157.csv', 's3_111.csv', 's3_260.csv', 's5_55.csv']
GP_TEST_VALIDATION_SET_SIZE = 2
GP_PLANO_VALIDATION_SET = [f'{s.split(".")[0]}.json' for s in GP_TEST_VALIDATION_SET]

GROZI_ROOT = utils.rel_path(*DATA_DIR, 'GroZi-120')

MODEL_DIR = ('..', 'models')
PRETRAINED_GAN_FILE = utils.rel_path(*MODEL_DIR, 'pretrained_dihe_gan.tar')
ENCODER_FILE = utils.rel_path(*MODEL_DIR, 'encoder.tar')

OUT_DIR = utils.rel_path('out')
