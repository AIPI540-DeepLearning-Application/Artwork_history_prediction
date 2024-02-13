import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score

class art_svm:
    '''
    This class is used to train a SVM model on the artwork dataset
    '''
    
    def __init__(self):
        self.data_folder = "../data_aug/train"
    
        
    def load_data(self, folder):
        images = []
        labels = []
        label_names = sorted(os.listdir(folder))
        for label_name in label_names:
            label_folder = os.path.join(folder, label_name)
            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, filename)
                    img = cv2.imread(img_path)
                    try:                
                        # convert the image to grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                        
                        # resize the image to 100x100
                        img = cv2.resize(img, (100, 100))
                        
                        # normalize the image
                        img = img / 255.0 
                        
                        # flatten the image to one dimension
                        images.append(img.flatten())
                        labels.append(label_name)
                    except:
                        print("Error reading file: ", img_path)
                        
        return np.array(images), np.array(labels)

    def split_data(self):
        X, y = self.load_data(self.data_folder)
        print("finished loading data")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def train(self):
        # train the svm model
        X_train, X_test, y_train, y_test = self.split_data()
        
        clf = svm.SVC(kernel='rbf')
        print("fitting the model")
        clf.fit(X_train, y_train)
        
        # predict on the test set
        y_pred = clf.predict(X_test)

        # print the classification report
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("accuracy:", accuracy)
        print("f1 score:", f1)
        print("\classification report:\n", classification_report(y_test, y_pred))

def main():
    model = art_svm()
    model.train()

if __name__ == '__main__':
    main()