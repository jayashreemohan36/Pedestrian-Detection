Pedestrian Detection using Machine Learning & Image Processing

Since Histogram of Oriented Gradients(HOG) is proven to be the best feature descriptor for Human Detection, I have employed HOG features trained on Support Vector Machine classifier to detect pedestrians.

The below diagram describes my implemtation workflow of Pedestrian Detection System.

![image](https://user-images.githubusercontent.com/75206889/111927112-d6a54f80-8aaf-11eb-8a26-b76fa7973335.png)

Training Stage: 

The SVM classifer is trained on INRIA dataset.
Principal Component Analysis(PCA) and HOG Hyperparamter tuning is carried out to increase the accuracy


Detection Stage:

Image Pyramid and sliding window technique are employed for every image frame from the video input to robust detection of pedestrians.


Detection Results:

![image](https://user-images.githubusercontent.com/75206889/111927417-dd809200-8ab0-11eb-9834-cb9fabd177b3.png) ![image](https://user-images.githubusercontent.com/75206889/111927483-233d5a80-8ab1-11eb-872b-15a9f5434843.png)




 
