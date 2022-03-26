import os
import sys
import glob
import time

import dlib

faces_folder = "../examples/faces"
# 用训练出的predictor进行实际应用
# # Now let's use it as you would in a normal application.  First we will load it
# # from disk. We also need to load a face detector to provide the initial
# # estimate of the facial location.
predictor = dlib.shape_predictor("results/predictor.dat")
detector = dlib.simple_object_detector("results/detector.svm")
#
# # Now let's run the detector and shape_predictor over the images in the faces
# # folder and display the results.
print("Showing detections and predictions on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

#     # Ask the detector to find the bounding boxes of each face. The 1 in the
#     # second argument indicates that we should upsample the image 1 time. This
#     # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    win.add_overlay(dets)
    time.sleep(2)