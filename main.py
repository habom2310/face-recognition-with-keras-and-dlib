import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import dlib
from model import create_model
from align import AlignDlib
import glob
import imutils


# INITIALIZE MODELS
nn4_small2 = create_model()

nn4_small2.summary()

nn4_small2.load_weights('weights/nn4.small2.v1.h5')

alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

#LOAD TRAINING INFORMATION
train_paths = glob.glob("image/*")
print(train_paths)

nb_classes = len(train_paths)

df_train = pd.DataFrame(columns=['image', 'label', 'name'])

for i,train_path in enumerate(train_paths):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[image,i,name]
        
print(df_train.head())

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_image(img):
    #print(img.shape)
    (h,w,c) = img.shape
    bb = dlib.rectangle(0, 0, w, h)
    #print(bb)
    return alignment.align(96, img, bb,landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
  
def load_and_align_images(filepaths, margin,image_size = 96):
    
    aligned_images = []
    for filepath in filepaths:
        #print(filepath)
        img = cv2.imread(filepath)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #aligned = resize(img, (image_size, image_size), mode='reflect')
        aligned = align_image(img)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
            
    return np.array(aligned_images)
    
def calc_embs(filepaths, margin=10, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = load_and_align_images(filepaths[start:start+batch_size], margin)
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)

    return np.array(embs)
    
def align_images(faces):
    aligned_images = []
    for face in faces:
        #print(face.shape)
        aligned = align_image(face)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
        
    return aligned_images

def calc_emb_test(faces):
    pd = []
    aligned_images = align_images(faces)
    if(len(faces)==1):
        pd.append(nn4_small2.predict_on_batch(aligned_images))
    elif(len(faces)>1):
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
      
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)
    return np.array(embs)

# TRAIN    
    
label2idx = []

for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))

train_embs = calc_embs(df_train.image)
np.save("train_embs.npy", train_embs)

train_embs = np.concatenate(train_embs)

# ANALYSE
import matplotlib.pyplot as plt

match_distances = []
for i in range(nb_classes):
    ids = label2idx[i]
    distances = []
    for j in range(len(ids) - 1):
        for k in range(j + 1, len(ids)):
            distances.append(distance.euclidean(train_embs[ids[j]].reshape(-1), train_embs[ids[k]].reshape(-1)))
    match_distances.extend(distances)
    
unmatch_distances = []
for i in range(nb_classes):
    ids = label2idx[i]
    distances = []
    for j in range(10):
        idx = np.random.randint(train_embs.shape[0])
        while idx in label2idx[i]:
            idx = np.random.randint(train_embs.shape[0])
        distances.append(distance.euclidean(train_embs[ids[np.random.randint(len(ids))]].reshape(-1), train_embs[idx].reshape(-1)))
    unmatch_distances.extend(distances)
    
_,_,_=plt.hist(match_distances,bins=100)
_,_,_=plt.hist(unmatch_distances,bins=100,fc=(1, 0, 0, 0.5))

plt.show()

threshold = 0.6 

# TEST
test_paths = glob.glob("test_image/*.jpg")
for path in test_paths:
    test_image = cv2.imread(path)
    show_image = test_image.copy()

    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(test_image, 0)
    
    faces = []
    
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        face = test_image[y1:y2,x1:x2]
        
        faces.append(face)

    print("len(faces) = {0}".format(len(faces)))
    if(len(faces)==0):
        print("no face detected!")
        continue
    else:    
        test_embs = calc_emb_test(faces)

    test_embs = np.concatenate(test_embs)
        
    people = []
    for i in range(test_embs.shape[0]):
        distances = []
        for j in range(len(train_paths)):
            distances.append(np.min([distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
            #for k in label2idx[j]:
              #print(distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)))
        if np.min(distances)>threshold:
            people.append("unknown")
        else:
            res = np.argsort(distances)[:1]
            people.append(res)

    names = []
    title = ""
    for p in people:
      if p == "unknown":
          name = "unknown"
      else:
          name = df_train[(df_train['label']==p[0])].name.iloc[0]
      names.append(name)
      title = title + name + " "
      
      
    for i,faceRect in enumerate(faceRects):
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        cv2.rectangle(show_image,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.putText(show_image,names[i],(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),3,cv2.LINE_AA)
        

    show_image = imutils.resize(show_image,width = 720)   
    cv2.imshow("result",show_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

