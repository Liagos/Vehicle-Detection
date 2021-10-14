import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from skimage import measure
import pafy
import os
import shutil
import time
import keyboard


class connectCam:
    def __init__(self):
        self.counter = 0
        self.timer = time.time()

    def connectCam(self, urlStream):
        video = pafy.new(urlStream)
        best = video.getbest(preftype='mp4')
        self.cap = cv.VideoCapture(best.url)

        if not self.cap.isOpened():
            print('can not open camera')
            exit()
        else:
            print("Connected to stream")

    def viewStream(self):

        self.ret, self.frame = self.cap.read()
        resized_frame = cv.resize(self.frame, (640, 360))
        # cv.imshow("Traffic Stream", resized_frame)

        return self.ret, self.frame, self.cap

    def createDir(self, outputPath):
        if os.path.exists(outputPath):
            print('output folder already exists: {}'.format(outputPath))
            print('do you want to delete this output folder: [y]')
            inp = input()
            if inp == 'y':
                shutil.rmtree(outputPath)
                time.sleep(0.5)
            else:
                print('script finished')
                exit()

        os.mkdir(outputPath)

        return outputPath

    def frameGrabbing(self, outputPath, frameRate):  # optional grab frame automatically
        if time.time() - self.timer > 1 / frameRate:
            # res_frame = cv.resize(self.frame, (640, 360))
            # cv.imshow("Frame Rate", res_frame)
            self.timer = time.time()
        try:
            if keyboard.is_pressed("s"):
                cv.imwrite(os.path.join(outputPath, 'frame_{}.png'.format(str(self.counter).zfill(6))), self.frame)
                print("Frame saved!")
                self.counter += 1
        except:
            pass

        return self.frame


class imageProcessing:
    def __init__(self):
        self.frame_list = []
        self.res = []
        self.medianFrame = []
        self.counter = 0

    def medianBackground(self, frame):
        # res_frame = cv.resize(frame,(640,360))
        self.frame_list.append(frame)
        if len(self.frame_list) > 2:
            self.frame_list.pop(0)
        array_from_list = np.array(self.frame_list)
        self.medianFrame = np.median(array_from_list[:, :, :, :], axis=0)

        return self.medianFrame

    def subBackground(self, frame, medFrame):
        diff = np.abs(frame - medFrame)
        diff = cv.cvtColor(np.uint8(diff) * 255, cv.COLOR_RGB2GRAY)
        resized_diff = cv.resize(diff, (640, 360))
        # cv.imshow("Diff", resized_diff)

        return diff


class imageFiltering:
    def __init__(self):
        self.counter = 0

    def blurImage(self, background_subtracted):
        blurry = cv.medianBlur(background_subtracted, 9)
        resized_blurry = cv.resize(blurry, (640, 360))
        # cv.imshow("Blurry", resized_blurry)

        return blurry

    def otsuThreshold(self, blurry_image):
        (T, threshInv) = cv.threshold(blurry_image, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        resized_thresholded = cv.resize(threshInv, (640, 360))
        # cv.imshow("Thresholded", cv.bitwise_not(resized_thresholded))

        return threshInv

    def K_means(self, background_subtraction):
        img_vectorised = background_subtraction.reshape(-1, 1)

        image_array_sample = shuffle(img_vectorised, random_state=0)[:2000]
        estimator = KMeans(n_clusters=2, random_state=0, max_iter=100)
        estimator.fit(image_array_sample)
        cluster_assignments = estimator.predict(img_vectorised)

        compressed_palette = estimator.cluster_centers_
        compressed_img = np.zeros(
            (background_subtraction.shape[0], background_subtraction.shape[1], compressed_palette.shape[1]))
        label_idx = 0
        for j in range(background_subtraction.shape[0]):
            for k in range(background_subtraction.shape[1]):
                compressed_img[j][k] = compressed_palette[cluster_assignments[label_idx]]
                label_idx += 1
                try:
                    if keyboard.is_pressed("b"):
                        cv.imwrite(os.path.join(outputPath, 'binary_{}.png'.format(str(self.counter).zfill(6))),
                                   compressed_img)
                        print("Binary saved!")
                        self.counter += 1
                except:
                    pass

        return compressed_img

    def morphologicalOperations(self, blurry_image):
        kernel = np.ones((7, 7), np.uint8)

        dilation = cv.dilate(blurry_image, kernel, iterations=2)
        opening = cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel)
        erosion = cv.erode(opening, kernel, iterations=2)
        dilation = cv.dilate(erosion, kernel, iterations=2)
        closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)

        resized_closing = cv.resize(closing, (640, 360))
        cv.imshow("Morphological Operations", cv.bitwise_not(resized_closing))

        return closing


class BBox:
    def __init__(self):
        self.coordinates = []
        self.padding = 10
        self.rect_extended = 0

    def canny_edge(self, morph_img, frame):
        edges = cv.Canny(morph_img, 30, 150)

        contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            self.coordinates.append([x, y, w, h])
        coordinates = np.array(self.coordinates)

        for coor in coordinates:
            padding_width = int(coor[2] * (self.padding / 100))
            padding_height = int(coor[3] * (self.padding / 100))

            rect_x = coor[0] - padding_width
            rect_y = coor[1] - padding_height
            rect_width = coor[2] + (2 * padding_width)
            rect_height = coor[3] + (2 * padding_height)

            if rect_x < 0:
                rect_x = 0
            if rect_y < 0:
                rect_y = 0
            if rect_width >= edges.shape[0]:
                rect_width = edges.shape[0]
            if rect_height >= edges.shape[1]:
                rect_height = edges.shape[1]

            tracker = cv.TrackerMIL_create()

            self.rect_extended = (rect_x, rect_y, rect_width, rect_height)

            tracker.init(frame, self.rect_extended)
            ret, box = tracker.update(frame)
            if ret:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv.imshow("RECT", frame)


if __name__ == '__main__':
    webCam = connectCam()
    imaging = imageProcessing()
    filtering = imageFiltering()
    bbox = BBox()

    # Connect to url and receive capture
    webCam.connectCam("https://www.youtube.com/watch?v=QTU6KUI1bf8")

    # Create Directory
    outputPath = webCam.createDir(outputPath="/Users/odysseas/Desktop/Frames")

    # Visualize stream
    while True:
        ret, frame, cap = webCam.viewStream()

        if not ret:
            print("Cannot receive frame")
            break

        # Hit [s] and grab a frame
        webCam.frameGrabbing(outputPath, frameRate=2)

        # Calcualte median frame
        medianFrame = imaging.medianBackground(frame)

        # Calculate difference
        bckg = imaging.subBackground(frame, medianFrame)

        # Blur image
        blurry_image = filtering.blurImage(bckg)

        # Threshold with Otsu or K_means
        thresholded = filtering.otsuThreshold(blurry_image)

        # Morphologial Operations
        morph_image = filtering.morphologicalOperations(thresholded)

        # Canny
        bbox.canny_edge(morph_image, frame)
        # cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Exit when keyboard hit [q]
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
