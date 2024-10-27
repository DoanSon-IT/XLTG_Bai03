import cv2
import os
import albumentations as A
import numpy as np

# Set the number of augmented images per original image
AUGMENTED_IMAGES_COUNT = 2

# Define the augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.CLAHE(p=0.3),
    A.RandomGamma(p=0.2),
])

# Function to apply augmentations and save images
def augment_and_save_images(folder):
    augmented_folder = f"{folder}_augmented"
    os.makedirs(augmented_folder, exist_ok=True)
    
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not open image {image_path}. Skipping.")
            continue
        
        for i in range(AUGMENTED_IMAGES_COUNT):
            augmented = transform(image=image)["image"]
            new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(augmented_folder, new_filename), augmented)
        print(f"Saved augmented images for {filename}.")

# Paths to original data folders
horse_folder = './Horse'
lion_folder = './Lion'

augment_and_save_images(horse_folder)
augment_and_save_images(lion_folder)
