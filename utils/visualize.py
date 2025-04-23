import os
import cv2
import glob
from tqdm import tqdm
def yolo_to_xyxy(bbox, img_w, img_h):
    x_center, y_center, width, height = bbox
        
    # Calculate absolute coordinates
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h
    
    # Convert to (x_min, y_min, x_max, y_max)
    x_min = int(x_center - width/2)
    y_min = int(y_center - height/2)
    x_max = int(x_center + width/2)
    y_max = int(y_center + height/2)

    return x_min, y_min, x_max, y_max

imgs= glob.glob(r'./filtered_data/test/images/*')
labels_path= r'./filtered_data/test/labels'
classes =  ['Hardhat', 'NO-Hardhat']
save_dir='vis'


os.makedirs(save_dir, exist_ok=True)
for img_path in tqdm(imgs):
    name= img_path.split('\\')[-1][:-4]
    img = cv2.imread(img_path)

    with open(labels_path +'/'+ name +'.txt', 'r') as f:
        lines= f.read().strip().split('\n')

    img_h, img_w = img.shape[:2]
    for line in lines:
        
        line= line.split(' ')
        cls_id = line[0]
        bbox= list(map(float, line[1:]))
        x_min, y_min, x_max, y_max= yolo_to_xyxy(bbox, img_w, img_h)
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (1,1,1), 3)
        cv2.putText(img, classes[int(cls_id)], (x_min, y_min-2), 0, 1, [255,255,255], 2)

    cv2.imwrite(save_dir+'/'+ name + '.jpg', img )
