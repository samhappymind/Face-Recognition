import cv2
import os
import numpy as np
from PIL import Image
from numpy import array
import imageio
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
import pickle
    
paths=['bhaskar','gopal','kavya','samrat']
 
def capture_sample_data():
    cascadeLocation='cascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadeLocation)
    size = 4
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("digital modernization AI")
    
    img_counter = 0
    
    while True:
        ret, img = cam.read()
        cv2.imshow("test", img)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                img_name = "capture_vid/User.6.{}.jpg".format(img_counter)
                cv2.imwrite(img_name, roi_gray)
                print("{} written!".format(img_name))
                img_counter += 1
    
    cam.release()    
    cv2.destroyAllWindows()
    
def prepare_dataset(directory):
   
    nbr=0
    images =[]
    labels=[]
   
    for path in paths:
        nbr =nbr + 1
        sub_directory = directory + '/' + path
        #cur_path = os.path.join(directory, path)
        
        image_paths = os.listdir(sub_directory)
        
        for im_path in image_paths:
            print(im_path)
            image_pil = cv2.imread(sub_directory+ '/'+ im_path)
            gray=cv2.cvtColor(image_pil,cv2.COLOR_BGR2GRAY);
            input_img_resize=cv2.resize(gray,(128,128))
            image = np.array(input_img_resize, 'uint8')
            images.append(image)
            labels.append(nbr)
            cv2.imshow("Reading Faces ",image)
            cv2.waitKey(10)
        
    return images,labels
    

def train_data():
    n_components = 256    
    pca = RandomizedPCA(n_components=n_components, whiten=True)
    clf=svm.SVC(kernel='rbf',C=5., gamma=0.001)
    
    train_directory = 'dataset/real_train'
    
    
    images, labels = prepare_dataset(train_directory)
    
    training_data=[]
    
    for i in range(len(images)):
        training_data.append(images[i].flatten())
    
    print("% shape of traing data => ",np.array(training_data).shape)
    
    print('labels =>',np.array(labels).shape)
   
    
    pca.fit(np.array(training_data))
    transformed = pca.transform(np.array(training_data))
    
    filename = 'models/pca_model.sav'
    pickle.dump(pca, open(filename, 'wb'))
    
    print("% shape of transformed data => ",transformed.shape)
    
    clf.fit(transformed,np.array(labels))
    
    filename = 'models/svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
def test_data(path):
    
    clf = pickle.load(open('models/svm_model.sav', 'rb'))
    pca = pickle.load(open('models/pca_model.sav', 'rb'))
    image_cv=cv2.imread(path)
    print(" img path is ",path)
    gray=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY);
    input_img_resize=cv2.resize(gray,(128,128))
    pred_image = np.array(input_img_resize, 'uint8')
    X_test = pca.transform(np.array(pred_image).flatten().reshape(1,-1))
    
    #print(X_test)
    
    mynbr = clf.predict(X_test)
    print ("Predicted By Classifier : ",mynbr[0])    
    print("predicted person is : ",paths[mynbr[0]-1])  


#train_data()
#E:/softwares/edurekaProjectWrkspace/faceRecognition/dataset/real_test/test.jpg
test_data('E:/softwares/edurekaProjectWrkspace/faceRecognition/dataset/real_test/test.jpg')
#capture_sample_data()
