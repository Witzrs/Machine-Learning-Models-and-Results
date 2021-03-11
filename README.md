Machine Learning Models and Results


Tensorflow Classification.py must be placed in the same directory as the Images2Classify folder.
The initial directory structure should be:


|
|
|__PyAudioAnalysis
|           |__knn
|           |__svm
|           |__forest
|           |__gradient
|           |__trees
|
|
|__Recordings
|           |__Species A
|           |          |__recording1.wav
|           |          |__recording2.wav
|           |          |__...
|           |
|           |__Species B
|           |          |__recording1.wav
|           |          |__recording2.wav
|           |          |__...
|           |__ ...
|
|__Tensorflow
|           |
|           |__Images2Classify/
|           |   |__images/
|           |   |__test.csv
|           |
|           |__object_detection.py
|           |__TensorFlow Classification.py
|           |__tf_statistical_analysis.py
|
|
|__spectrogrammer.py

TensorFlow Classification.py is used to draw the detections 