import os
import glob
import shutil
from tqdm import tqdm 


def Filter_and_Clean(images_path, labels_path, classes, save_dir):
    """
    Filter images based on given classes , remove all the empty images and corrupted files
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'/labels', exist_ok=True)
    os.makedirs(save_dir+'/images', exist_ok=True)

    new_ids = {cls_id : str(i) for i, cls_id in enumerate(classes)}

    for txt_file in tqdm(os.listdir(labels_path)):
        new_lines=[]
        name= txt_file.split('/')[-1][:-4]
        try:
            with open(labels_path+ '/'+ txt_file, 'r') as f:
                lines= f.read().strip().split('\n')
        except:
            # print('There is an issue with file: ', name)
            continue

        for line in lines :
            if line!= '':
                cls_id= line.split(' ')[0]
                if cls_id in classes:
                    new_line= new_ids[cls_id] + line[1:]
                    new_lines.append(new_line)

        if new_lines != []: # save only images that contains the given classes
            img_path = images_path+'/'+name+ '.jpg'

            if os.path.exists(img_path):
                shutil.copyfile(img_path, save_dir+'/images/'+name+ '.jpg' )
            else :
                print('image not found: ', name)
                continue

            with open(save_dir + '/labels/'+ name+'.txt', 'w') as f:
                f.write('\n'.join(new_lines))



data_path= './dataset/raw_data'
new_path = './filtered_data'
chosen_classes= ['3', '8']

Filter_and_Clean(images_path= data_path+ '/train/images',
                 labels_path= data_path+ '/train/labels',
                 classes=chosen_classes,  save_dir= new_path+ '/train' )

Filter_and_Clean(images_path= data_path+ '/test/images',
                 labels_path= data_path+ '/test/labels',
                 classes=chosen_classes,  save_dir= new_path+ '/test' )

Filter_and_Clean(images_path= data_path+ '/valid/images',
                 labels_path= data_path+ '/valid/labels',
                 classes=chosen_classes,  save_dir= new_path+ '/valid' )
