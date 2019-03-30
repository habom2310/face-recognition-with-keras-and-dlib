import cv2
import glob
import pandas as pd
from imageio import imread,imsave
from skimage.transform import resize
from tqdm import tqdm
import dlib


train_paths = glob.glob("image/*")
print(train_paths)

df_train = pd.DataFrame(columns=['image', 'label', 'name'])

for i,train_path in tqdm(enumerate(train_paths)):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[image,i,name]
        
print(df_train)
        
for img_path in df_train.image:
    print(img_path)
    image = imread(img_path)
    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(image, 0)

    faceRect = faceRects[0]
    if (faceRect==None):
        continue

    x1 = faceRect.left()
    y1 = faceRect.top()
    x2 = faceRect.right()
    y2 = faceRect.bottom()

    face = image[y1:y2,x1:x2]
    imsave(img_path,face)
    