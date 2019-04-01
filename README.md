# Face recognition with keras and dlib
In this example, Keras is used to implement CNN model inspired by [OpenFace project](http://cmusatyalab.github.io/openface/). The model is a variant of the NN4 architecture and identified as nn4.small2 model in the OpenFace project. The model training aims to learn an embedding of an image that the L2 distance between all faces of the same identity is small and the distance between a pair of faces from different identities is large. By selecting suitable threshold, the model can recognize faces in own dataset. Note that this model can run on CPU.

<img src="https://github.com/habom2310/face-recognition-with-keras-and-dlib/blob/master/result/res1.JPG" width="270"/> <img src="https://github.com/habom2310/face-recognition-with-keras-and-dlib/blob/master/result/res2.JPG" width="270"/> <img src = "https://github.com/habom2310/face-recognition-with-keras-and-dlib/blob/master/result/res3.JPG" width="270">

# Pre-requisite
 ## Install needed packages
 - keras, dlib (can use the wheel file for windows)
 - scikit-learn, tqdm, pandas
 ## Pre-trained models
 - Download `shape_predictor_68_face_landmarks.dat` from [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) and put it in the project folder.
 
# Implement
 ## Prepare images for training

- In the folder `image` there are folders containing images of people that we want to recognize. Each folder has 5 images of a person. If you want to have more people, just create folders and put images inside. It is recommended to have at least 5 images per person and the number of images of each person should be equal. 

- The images used for training should have only ONE face of the person.

- Image file must be in `.jpg` format. 

- Run `face_detect_and_save.py`. It will go through images in folders in the `image` and detect the face and save it (replace the full image).

 ## Training and testing
 - Run `main.py`. It will do following things: 
   - Initialize models.
   - Load images for training.
   - Start training and save `train.embs` as the output (for instant use without training later).
   - Plot a graph showing the difference in the distance between match and unmatch training images. The value of threshold can be chosen to separate match and unmatch faces. Modify the `threshold` in line 158 in `main.py`.
   ![alt-text](https://github.com/habom2310/face-recognition-with-keras-and-dlib/blob/master/result/graph.JPG)
   - Test with images in the `test_image` folder. You can make this part a separate .py file, just need to load the `train.embs`.
   - Images for testing can have multiple faces. Faces from people not in the training data will be shown as "unknown".
   
# Reference
- [Deep face recognition with Keras, Dlib and OpenCV](https://krasserm.github.io/2018/02/07/deep-face-recognition/)
