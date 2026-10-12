import numpy as np
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class Cut(ImageOnlyTransform):
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    
    def apply(self, image, **params):
        
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "cutting")


def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    
    
    satellite_transforms = A.Compose([

                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    return satellite_transforms, ground_transforms



def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,uncertain_type = 'None',p=1,strenth=(57, 59)):
    print('uncertain_type:',uncertain_type)
    if uncertain_type == 'None':
        satellite_transforms = A.Compose([
                                          A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                          A.Normalize(mean, std),
                                          ToTensorV2(),
                                         ])
    
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                       A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                       A.Normalize(mean, std),
                                       ToTensorV2(),
                                      ])
    elif uncertain_type=='blur':
        print(f'blur_limit=={strenth}','p=',p)
        satellite_transforms = A.Compose([
                                          A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                          A.GaussianBlur(sigma_limit=strenth,p=p),
                                          A.Normalize(mean, std),
                                          ToTensorV2(),
                                         ])
    
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                       A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                       A.GaussianBlur(sigma_limit=strenth,p=p),
                                       A.Normalize(mean, std),
                                       ToTensorV2(),
                                      ])
    elif uncertain_type=='noise':
        noise_limit=strenth
        print(f'noise_limit=={noise_limit}','p=',p)
        satellite_transforms = A.Compose([
                                          A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                          A.ShotNoise(scale_range=noise_limit,p=p),
                                          A.Normalize(mean, std),
                                          ToTensorV2(),
                                         ])
    #var_limit ShotNoise
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                       A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                       A.ShotNoise(scale_range=noise_limit,p=p),
                                       A.Normalize(mean, std),
                                       ToTensorV2(),
                                      ])
    elif uncertain_type == 'brightness':
        beta=strenth
        print(f'brightness=={beta}','p=',p)
        satellite_transforms = A.Compose([
            A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
            A.RandomBrightnessContrast(
            brightness_limit=beta,
            contrast_limit=0,
            brightness_by_max=False,
            p=p
        ),
            A.Normalize(mean, std),
            ToTensorV2(),
        ])
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                       A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                       A.RandomBrightnessContrast(
            brightness_limit=beta,
            contrast_limit=0,
            brightness_by_max=False,
            p=p
        ),
                                       A.Normalize(mean, std),
                                       ToTensorV2(),
                                      ])

    return satellite_transforms, ground_transforms