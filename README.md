# Vehicle Detection

This repository contains the code for the project "**Vehicle Detection**". The goal of this project is to detect vehicles on a highway from real-time video frames using image processing techniques.

The repository is organised as follows:
- The `Traffic_PyCharm_V2.py` file opens the camera stream and detects vehciles.
- The `args.py` file parses the arguments needed for the execution.
- The `confing.ini` file contains the camera configuration, frame rate, and maximum number of files to save in each folder.

To run the code, you will need to install the libraries in the `requirements.txt` file. We recommend using [PyCharm](https://www.jetbrains.com/pycharm/promo/?source=google&medium=cpc&campaign=14123077402&term=pycharm&gclid=Cj0KCQjw6_CYBhDjARIsABnuSzqkMV4IXzjuVu-enSX0e70lwTUQBmgEFAoSE3uktD045-LG9A0s0acaAqEDEALw_wcB).

To run an example, execute the following command:

`python Traffic_PyCharm_V2.py camera_x`

where `x` is the ID of the camera you want to use (from 1 to 6).

The available cameras are the following:

- [camera_1](https://www.youtube.com/watch?v=69Q7I4YQVj0) N233 brug Rhenen, Kesteren live HD PTZ

# 318 Aalten live HD PTZ
camera_2 = https://www.youtube.com/watch?v=Su5bUPT5_04

# N332 Lochem live HD PTZ
camera_3 = https://www.youtube.com/watch?v=j3yBBXNct9M

# N348a Den Elterweg, Zutphen live HD PTZ
camera_4 = https://www.youtube.com/watch?v=keIFkcf6B5k

# N302 Aquaduct Harderwijk live ultraHD PTZ
camera_5 = https://www.youtube.com/watch?v=Sex3fwYwQ0w

# N325b Arnhem HD PTZ
camera_6 = https://www.youtube.com/watch?v=Sk0aQxTygxo
