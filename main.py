import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of each person
    '''
    
    return os.listdir(root_path)

def get_test_names(root_path):
    '''
        To get a list of file (perosn) names from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of each person
    '''
    names = list()
    
    for filename in os.listdir(root_path):
        names.append(filename.split('_')[0])
    
    return names
    
    

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    
    images_list = list()
    classes_list = list()
    for index, class_name in enumerate(train_names):
        
        class_path = root_path + '/' + class_name
        images_path_list = [ class_path + '/' + filename for filename in os.listdir(class_path)]
        
        for image_path in images_path_list:
            img = cv2.imread(image_path)
            images_list.append(img)
            classes_list.append(index)
    
    return images_list,classes_list


def detect_train_faces_and_filter(image_list, image_classes_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered image classes id
    '''
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    filtered_images_list = list()
    filtered_classes_list = list()
    
    
    for index, image in enumerate(image_list):
        
        # # preview the image
        # plt.title('Before Grayscaled')
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB ))
        # plt.show()
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # # preview the grayscaled image
        # plt.title('After Grayscaled')
        # plt.imshow(image_gray, cmap='gray')
        # plt.show()
        
        detected_faces = face_cascade.detectMultiScale(image_gray,scaleFactor=1.2,minNeighbors=5)
        
        if len(detected_faces) < 1:
            # # show the image that has no face detected
            # plt.title('No Faces Detected')
            # plt.imshow(image_gray, cmap='gray')
            # plt.show()
            continue
        
        for (x,y,w,h) in detected_faces:
            
            # # draw the detected face
            # temp_image_gray = image_gray
            
            # cv2.rectangle(temp_image_gray,(x,y),(x+w,y+h),(255),2)
            
            # plt.title('Detected Faces')
            # plt.imshow(temp_image_gray, cmap='gray')
            # plt.show()
            
            cropped_faces = image_gray[y:y+h, x:x+w]
            
            
            
            eyes = eye_cascade.detectMultiScale(cropped_faces)
            
            #face is detected but not the eye, so it's most probably there's a mistake with the face cascade.
            if( len(eyes) < 1):
                
                
                # # show the face with no eyes detected
                # plt.title('Face Detected, But No Eyes Not Detected: ')
                # plt.imshow(cropped_faces, cmap='gray')
                # plt.show()
                continue
            else:
                
                # # draw and show the detected eyes
                # temp_cropped_faces = cropped_faces
                # for (ex,ey,ew,eh) in eyes:
                #     cv2.rectangle(temp_cropped_faces,(ex,ey),(ex+ew,ey+eh),(255),2)
                
                # plt.title('Face and Eyes Detected: ')
                # plt.imshow(temp_cropped_faces, cmap='gray')
                # plt.show()
                
                # # showing cropped face 
                # plt.title(train_names[image_classes_list[index]])
                # plt.imshow(cropped_faces, cmap='gray')
                # plt.show()
                
                # only save the image to the list if there's an detected
                
                
                filtered_images_list.append(cropped_faces)
                filtered_classes_list.append(image_classes_list[index])
            
            
            
    return filtered_images_list, filtered_classes_list
    

def detect_test_faces_and_filter(image_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
    '''
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    filtered_images_list = list()
    filtered_faces_rectangle = list()
    
    
    for index, image in enumerate(image_list):
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detected_faces = face_cascade.detectMultiScale(image_gray,scaleFactor=1.2,minNeighbors=5)
        
        if len(detected_faces) < 1:
            continue
        
        for (x,y,w,h) in detected_faces:
            cropped_faces = image_gray[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(cropped_faces)
        
            #face is detected but not the eye, so it's most probably there's a mistake with the face cascade.
            if( len(eyes) < 1):
                continue
            else:
                
                #only save the image to the list if there's an detected
                filtered_images_list.append(cropped_faces)
                filtered_faces_rectangle.append((x,y,w,h))
        
            
    return filtered_images_list, filtered_faces_rectangle
    
    
    

def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))
    
    return face_recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image in the test directories
    '''
    images_list = list()
    
    images_path_list = [ test_root_path + '/' + filename for filename in os.listdir(test_root_path)]
    
    for image_path in images_path_list:
        img = cv2.imread(image_path)
        images_list.append(img)
    
    return images_list
    
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        test_faces_gray : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    prediction_results = list()
    
    for test_face in test_faces_gray:
        
        prediction, _ = recognizer.predict(test_face)
        
        prediction_results.append(prediction)
    
    return prediction_results
        

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, size):
    '''
        To draw prediction results on the given test images and resize the image

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        size : number
            Final size of each test image

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    
    drawn_images = list()
    
    for index, test_image in enumerate(test_image_list):
        
        x,y,h,w = test_faces_rects[index]
        
        #draw rectangle around the face
        cv2.rectangle(test_image, pt1= (x,y), pt2 = (x+w , y+h), color = (0,255,255), thickness = 2 )
        
        #resize
        resized_test_image = cv2.resize(test_image, dsize = (size,size), interpolation = cv2.INTER_AREA)
        
        #write the name on image top-left
        label = 'Prediction: ' + train_names[predict_results[index]]
        
        cv2.putText(resized_test_image, text = label, org= (10,20), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color = (0,0,255), thickness = 1)
        
        actual = 'Actual: ' + test_names[index]
        
        cv2.putText(resized_test_image, text = actual, org= (10,180), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color = (255,0,0), thickness = 1)
        
        drawn_images.append(resized_test_image)
        
        
    return drawn_images
        
def combine_and_show_result(image_list, size):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
        size : number
            Final size of each test image
    '''
    
    for index, image in enumerate(image_list):
        if(index == 0):
            result = image
        else:
            result = np.hstack( (result,image) )
        
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyWindow('result')

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    test_names = get_test_names(test_root_path)
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, 200)
    
    combine_and_show_result(predicted_test_image_list, 200)