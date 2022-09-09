import os
import pafy
import time
import shutil
import argparse
import cv2 as cv
import numpy as np
import skimage.measure
import skimage.feature
import skimage.morphology
from args import arguments
from removeShadow import shadow
# from yolo_detector import yoloV5
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from datetime import date, datetime
from configparser import ConfigParser


class connectCam:
    def __init__(self):
        self.counter = 0
        self.timer = time.time()
        self.local_time = time.localtime()

    def connectCam(self, urlStream):
        video = pafy.new(urlStream)
        best = video.getbest(preftype='mp4')
        self.cap = cv.VideoCapture(best.url)

        if not self.cap.isOpened():
            print('can not open camera')
            exit()
        # else:
        #     print("Connected to stream")

    def viewStream(self):

        self.ret, self.frame = self.cap.read()
        if self.ret:
            self.frame = cv.resize(self.frame, (640, 360))
            # cv.imshow("Traffic Stream", self.frame)

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
            res_frame = cv.resize(self.frame, (640, 360))
            #    cv.imshow("Frame Rate", res_frame)
            self.timer = time.time()
        # try:
        #    if keyboard.is_pressed("s"):
        #        cv.imwrite(os.path.join(outputPath, 'frame_{}.png'.format(str(self.counter).zfill(6))), self.frame)
        #        print("Frame saved!")
        #        self.counter += 1
        # except:
        #    pass
        #
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
        if len(self.frame_list) > 5:
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

    def gaussianBackground(self, frame):
        se1 = skimage.morphology.disk(5)
        se2 = skimage.morphology.square(5)
        mask = gaussianBack.apply(frame)
        blur_mask = cv.medianBlur(mask, 9)
        # open = cv.morphologyEx(mask, cv.MORPH_OPEN, se1)
        dil1 = cv.dilate(blur_mask, se2, iterations=2)
        dil2 = cv.dilate(dil1, se1, iterations=1)
        # cv.imshow("Gaussian Back", dil2)
        return dil2, blur_mask


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
                # try:
                #     if keyboard.is_pressed("b"):
                #         cv.imwrite(os.path.join(outputPath, 'binary_{}.png'.format(str(self.counter).zfill(6))),
                #                    compressed_img)
                #         print("Binary saved!")
                #         self.counter += 1
                # except:
                #     pass

        return compressed_img

    def morphologicalOperations(self, blurry_image):
        kernel = np.ones((7, 7), np.uint8)

        dilation = cv.dilate(blurry_image, kernel, iterations=2)
        opening = cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel)
        erosion = cv.erode(opening, kernel, iterations=2)
        dilation = cv.dilate(erosion, kernel, iterations=2)
        closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
        resized_closing = cv.resize(closing, (640, 360))
        # cv.imshow("Morphological Operations", cv.bitwise_not(resized_closing))

        return closing


class boundingBox:
    def __init__(self):
        self.coordinates = []
        self.padding = 10
        self.rect_extended = []

        self.comp_ext = []
        self.new_list = []

    def canny_edge(self, morph_img):
        edges = cv.Canny(morph_img, 30, 120)
        contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv.contourArea(c)
            if area > 200:
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

            self.rect_extended.append([rect_x, rect_y, rect_width, rect_height])
            if len(self.rect_extended) > 3:
                self.rect_extended.pop(0)

        return self.rect_extended

    def connectedComponents(self, frame, foreground):
        labels = skimage.measure.label(frame)
        regions = skimage.measure.regionprops(labels)

        contours, _ = cv.findContours(foreground, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        large_regions = [r for r in regions if r.area > 500]
        self.comp_ext.clear()
        for r in large_regions:
            (min_row, min_col, max_row, max_col) = r.bbox
            self.comp_ext.append([min_row, min_col, max_row, max_col])
        ccomp = np.array(self.comp_ext)

        self.comp_ext.clear()

        for idx, c in enumerate(ccomp):
            width = c[3] - c[1]
            height = c[2] - c[0]

            padding_height = int(height * (self.padding / 100))
            padding_width = int(width * (self.padding / 100))

            x = c[1] - padding_width
            y = c[0] - padding_height

            bbox_height = height + (2 * padding_height)
            bbox_width = width + (2 * padding_width)

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if bbox_width > frame.shape[0]:
                bbox_width = frame.shape[0]
            if bbox_height > frame.shape[1]:
                bbox_height = frame.shape[1]

            self.comp_ext.append([x, y, bbox_width, bbox_height])

        return self.comp_ext, contours


class saveFrame:
    def __init__(self):
        self.image_counter = 0

    def saveFramesFromStream(self, outputPath, frame, t):
        folder_length = os.listdir(outputPath)
        if len(folder_length) < parser.maxFolders:
            subfolder = str(outputPath) + "/" + str(today.strftime(parser.folderName)) + "_" + str(
                time.strftime("%H_%M_%S", t))
            if os.path.exists(subfolder):
                subfolder_length = os.listdir(subfolder)
                if len(subfolder_length) < parser.maxImages:
                    cv.imwrite(os.path.join(subfolder, parser.fileName.format(str(self.image_counter).zfill(6))),
                               frame)
                    self.image_counter += 1
                else:
                    t = time.localtime()
                    self.image_counter = 0
            else:
                os.mkdir(subfolder)

        return t, self.image_counter, subfolder


class saveAnnotations:

    def __init__(self):
        pass

    def annotation_bbox(self, rectangles, counter, subfolderName):
        f_bbox = open(subfolderName + "/" + "annotation_bbox.csv", "a+")

        if counter == 0:
            f_bbox.writelines(["# Annotation File", "\n"
                                    "# Format: image name, label, shape, ..ShapeAttributes.. ","\n"
                                    "#", "\n"
                                    "# Shape Attributes:", "\n"
                                    "# - rect: topLeftX, topLeftY, width, height", "\n"
                                    "# - point: x, y", "\n"
                                    "AnnotationFormatVersion:2.1", "\n", "\n"])

        for i in range(len(rectangles)):
            f_bbox.write(f"{counter}.png:rot0,nan,,rect,{rectangles[i][0]},"
                         f"{rectangles[i][1]},"
                         f"{rectangles[i][2]},"
                         f"{rectangles[i][3]}\n")

    def annotation_polygon(self, polygons, counter, subfolderName):
        f_poly = open(subfolderName + "/" + "annotation_poly.csv", "a+")
        if counter == 0:
            f_poly.writelines(["# Annotation File", "\n"
                               "# Format: image name, label, shape, ..ShapeAttributes.. ", "\n"
                               "#", "\n"
                               "# Shape Attributes:", "\n"
                               "# - rect: topLeftX, topLeftY, width, height", "\n"
                               "# - point: x, y", "\n"
                               "AnnotationFormatVersion:2.1", "\n", "\n"])

        for i in range(len(polygons)):
            flat_rect = polygons[i].flatten()
            converted_list = [str(element) for element in flat_rect]
            joined_string = ",".join(converted_list)
            f_poly.write(f"{counter}.png:rot0,nan,,polygon,{joined_string}\n")

    def imageList(self, counter, subfolderName):
        f_image_list = open(subfolderName + "/" + "imageNamesList.txt", "a+")
        f_image_list.write(parser.fileName.format(str(counter).zfill(6)) + "\n")


if __name__ == '__main__':
    webCam = connectCam()
    imaging = imageProcessing()
    filtering = imageFiltering()
    bbox = boundingBox()
    config_parser = arguments()
    parser = config_parser.getArgs()
    save_frame = saveFrame()
    save_annotations = saveAnnotations()
    remove_shadow = shadow()
    gaussianBack = cv.createBackgroundSubtractorMOG2(120, 50, detectShadows=False)

    today = date.today()
    local_time = time.localtime()

    # Connect to url and receive capture
    webCam.connectCam(parser.cameraUrl)

    # Create Directory
    outputPath = webCam.createDir(outputPath=parser.folderPath)

    # Visualize stream
    while True:
        ret, frame, cap = webCam.viewStream()

        if not ret:
            print("Cannot receive frame")
            break
        # Hit [s] and grab a frame
        # webCam.frameGrabbing(outputPath, frameRate=parser.frameRate)

        # Calculate median frame
        medianFrame = imaging.medianBackground(frame)

        # Calculate difference
        bckg = imaging.subBackground(frame, medianFrame)

        # Blur image
        blurry_image = filtering.blurImage(bckg)

        # Threshold with Otsu or K_means
        thresholded = filtering.otsuThreshold(blurry_image)
        # filtering.K_means(bckg)

        # Morphological Operations
        morph_image = filtering.morphologicalOperations(thresholded)

        # Canny
        rect = bbox.canny_edge(morph_image)
        # if rect:
        #     new_rect = np.array(rect)
        #     for i in range(new_rect.shape[0]):
        #         cv.rectangle(frame, (new_rect[i][0], new_rect[i][1]),
        #                      (new_rect[i][2] + new_rect[i][0], new_rect[i][3] + new_rect[i][1]),
        #                      (0, 255, 0), 2)

        # cv.imshow("Canny", frame)

        # Save frames and update local time
        local_time, image_counter, subfolderPath = save_frame.saveFramesFromStream(outputPath=parser.folderPath, frame=frame,
                                                                    t=local_time)

        # MOG Background Subtraction
        mog, mask_mog = imaging.gaussianBackground(frame)
        components, contours = bbox.connectedComponents(mog, mask_mog)
        # if contours:
        #     for c in range(len(contours)):
        #         area = cv.contourArea(contours[c])
        #         if area > 500:
        #         cv.drawContours(frame, contours, c, (0, 255, 0), 2)

        # Car Shadow Remove
        # carShadow, shadow = remove_shadow.removeCarShadow(frame, medianFrame, mask_mog)
        # cv.imshow("Detected shadow", carShadow)
        # cv.imshow("Car without shadow", shadow)

        # Drawing of BBox
        if components:
            new_comp = np.array(components)
            for i in range(new_comp.shape[0]):
                cv.rectangle(frame, (new_comp[i][0], new_comp[i][1]), (new_comp[i][2] + new_comp[i][0],
                                                                       new_comp[i][3] + new_comp[i][1]), (0, 255, 255), 2)

        cv.imshow("MOG", frame)


        # Save bbox and polygons in .txt files
        save_annotations.annotation_bbox(components, image_counter, subfolderPath)
        save_annotations.annotation_polygon(contours, image_counter, subfolderPath)
        save_annotations.imageList(image_counter, subfolderPath)

        # Exit when keyboard hit [q]
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
