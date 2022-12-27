# amogous
CS445 Final Project

Human face detection from images/videos and morphing into the imposter from Among Us.

A demo of the face detection can be found in the face_detection.ipynb file and is self contained.

All software depencancies can be installed using the requirements.txt file.

The data we used for testing can be found at https://huggingface.co/datasets/wider_face. These files are placed into the 'data' folder.

Basic demonstrations of the face morphing code can be found in face_morph.ipynb. To use this notebook, run the first four cells. Each subsequent section should have these first four cells run before execution each time. The calculations used can be found in affine_transform.py, and amongus.py.

To create a face morph using your webcam, run the webcam.py file. This will output the video to output/final_output.mp4. The code for this can be found in video_result.py and webcam.py.
