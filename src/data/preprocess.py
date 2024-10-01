import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def normalize_image(image):
    return (image - np.mean(image)) / np.std(image)

def preprocess_dataset(input_dir, output_dir, test_size=0.2):
    image_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'masks')
    
    images = []
    masks = []
    
    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        if os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            img = apply_clahe(img)
            img = normalize_image(img)
            
            images.append(img)
            masks.append(mask)
    
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=42)
    
    for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'masks'), exist_ok=True)
        
        for i, (img, mask) in enumerate(zip(X, y)):
            cv2.imwrite(os.path.join(split_dir, 'images', f'{i}.png'), img)
            cv2.imwrite(os.path.join(split_dir, 'masks', f'{i}.png'), mask)

if __name__ == '__main__':
    input_dir = 'data/raw'
    output_dir = 'data/processed'
    preprocess_dataset(input_dir, output_dir)