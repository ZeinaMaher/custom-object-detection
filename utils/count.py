import glob
from tqdm import tqdm 

labels= glob.glob('./filtered_data/test/labels/*')

classes={}
empty=0
num_img_per_class={}
for txt_file in tqdm(labels):
    with open(txt_file, 'r') as f:
        lines= f.read().strip().split('\n')

    flags= {}

    if lines ==[""]:
        empty+=1
        continue
    for line in lines :
        if line!= ['']:
            cls_id= line.split(' ')[0]
            try:
                classes[cls_id]+=1
            except:
                classes[cls_id]=0
            flags[cls_id] =True
    
    for cls_id in flags.keys():
        try:
            num_img_per_class[cls_id]+=1
        except:
            num_img_per_class[cls_id]=0
        

print('===================================================')
print('# of all images :', len(labels))
print('# of empty images :', empty)
print()
print('Classes Distribution: ')
print("num of images per class:", num_img_per_class)
print('# of instances per class:' ,classes)
