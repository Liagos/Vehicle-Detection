from configparser import ConfigParser
import argparse
from datetime import date, datetime
import time


class arguments:
    def __init__(self):
        self.file = 'config.ini'
        self.p = argparse.ArgumentParser()
        self.local_time = time.localtime()
        self.image_counter = 0
        self.today = date.today()
        self.subfolder = str("/Users/odysseas/Desktop/Frames") + "/" + str(self.today.strftime('%b_%d_%Y')) + "_" \
                         + str(time.strftime("%H_%M_%S", self.local_time))

    def getArgs(self):
        self.p.add_argument("-c", "--configFile", dest='config_file', default='config.ini', type=str,
                            help="path to config file")  # ,required=True)
        args = self.p.parse_args()

        config_file = args.config_file
        config = ConfigParser()
        config.read(config_file)

        print("please select camera, camera_1 to camera_6")
        inp = input()
        if not inp:
            inp = "camera_1"

        self.p.add_argument("-u", "--cameraURL", dest="cameraUrl", default=config["cameras"][inp], type=str,
                            help="select from camera 1-6, e.g. camera_1", choices=["camera_1", "camera_2", "camera_3",
                                                                                   "camera_4", "camera_5", "camera_6"])
        self.p.add_argument("-p", "--folderPath", dest="folderPath", default=config['folderPath']['path'], type=str,
                            help='folder path to save frames')
        self.p.add_argument("-r", "--frameRate", dest="frameRate", default=config["frameRate"]["frame_rate"], type=int,
                            help="frame rate of image capturing in frames per second")
        self.p.add_argument('-mf', '--maxFolders', dest="maxFolders", default=config["maxFolders"]["folder_number"],
                            type=int, help="maximum number of folders to create")
        self.p.add_argument('-nf', '--folderName', dest='folderName', default=config['folderName']['folder_name'],
                            type=str,
                            help="folder name")
        self.p.add_argument('-mi', '--maxImages', dest="maxImages", default=config["maxImages"]["images_number"],
                            type=int,
                            help="maximum number of images to record")
        self.p.add_argument('-fn', '--fileName', dest='fileName', default=config['fileName']['file_name'], type=str,
                            help="name of frame recorded")
        self.p.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

        args = self.p.parse_args()

        return args
