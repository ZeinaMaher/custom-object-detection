import numpy as np
import cv2
import random
import glob
from PIL import Image

def visualize(img, labels, output_path):
    # Rescale the normalized bounding box coordinates to pixel values
    h, w, _ = img.shape

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)


    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for label in labels:
        class_id, cx, cy, w_box, h_box = label
        x1 = int((cx - w_box / 2) * w)
        y1 = int((cy - h_box / 2) * h)
        x2 = int((cx + w_box / 2) * w)
        y2 = int((cy + h_box / 2) * h)

        # Draw bounding box
        print(type(img))
        print(img.shape)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(img, f"Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img )

class CustomAugmentations:
    def __init__(self,p_flip=0.5):
        self.p_flip = p_flip

    def __call__(self, img, labels):
        img, labels = self.random_horizontal_flip(img, labels)
        img, labels = self.adjust_brightness(img, labels)
        img, labels = self.adjust_contrast(img, labels)
        return img, labels

    def random_horizontal_flip(self, img, labels):
        """
        Horizontally flips the image and adjusts bounding boxes accordingly.

        Args:
            img (np.ndarray): Image in CHW format.
            labels (np.ndarray): Bounding boxes in YOLO format (class, cx, cy, w, h).
            p (float): Probability of applying the flip.

        Returns:
            img (np.ndarray): Flipped or original image.
            labels (np.ndarray): Adjusted or original labels.
        """
        if random.random() < self.p_flip:
            img = np.flip(img, axis=1).copy()  # Flip horizontally (flip on width)
            img =  np.ascontiguousarray(img)
            # Flip the label's cx coordinate
            if labels.size > 0:
                labels[:, 1] = 1 - labels[:, 1]  # Flip cx

        return img, labels

    def adjust_brightness(self, img, labels, factor_range=(0.6, 1.4)):
        """
        Adjusts the brightness of the image by a random factor.

        Args:
            img (np.ndarray): Image in CHW format.
            labels (np.ndarray): Bounding boxes (unchanged).
            factor_range (tuple): Range of brightness scaling factors.

        Returns:
            img (np.ndarray): Brightness-adjusted image.
            labels (np.ndarray): Unchanged labels.
        """
        factor = random.uniform(*factor_range)
        img = img * factor
        return img, labels

    def adjust_contrast(self,img, labels, factor_range=(0.6, 1.4)):
        """
        Adjusts the contrast of the image by a random factor.

        Args:
            img (np.ndarray): Image in CHW format.
            labels (np.ndarray): Bounding boxes (unchanged).
            factor_range (tuple): Range of contrast scaling factors.

        Returns:
            img (np.ndarray): Contrast-adjusted image.
            labels (np.ndarray): Unchanged labels.
        """
        factor = random.uniform(*factor_range)
        mean = img.mean(axis=(1, 2), keepdims=True)
        img = (img - mean) * factor + mean
        return img, labels

        
if __name__ == "__main__":
    imgs= glob.glob(r'C:\Users\Zeina Abu Ruqaia\Desktop\projects\Custom_Object_Detector\dataset\samples\valid\images\*')
    labels_path =r'C:\Users\Zeina Abu Ruqaia\Desktop\projects\Custom_Object_Detector\dataset\samples\valid\labels'
    output_path=r'C:\Users\Zeina Abu Ruqaia\Desktop\projects\Custom_Object_Detector\vis'

    augmentations = CustomAugmentations(p_flip=1.0)  # Set p_flip=1.0 to always flip for testing
    for img_path in imgs :
        name= img_path.split('\\')[-1][:-4]

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        # if img.dtype != np.uint8:
        #     img = img.astype(np.uint8) 
        print(img.shape)

        with open(labels_path +'//'+ name +'.txt', 'r') as f:
            lines= f.read().strip().split('\n')

        labels = [list(map(float, line.split(' '))) for line in lines]
        labels = np.array(labels) if labels else np.zeros((0, 5))  # Empty array if no labels    

        # Apply the augmentation
        augmented_img, augmented_labels = augmentations(img, labels)
        visualize(augmented_img, augmented_labels,output_path+'//'+ name + '.jpg')