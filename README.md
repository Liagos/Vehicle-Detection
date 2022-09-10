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

- [Camera 1](https://www.youtube.com/watch?v=69Q7I4YQVj0) (N233 brug Rhenen, Kesteren live HD PTZ)
- [Camera 2](https://www.youtube.com/watch?v=Su5bUPT5_04) (318 Aalten live HD PTZ)
- [Camera 3](https://www.youtube.com/watch?v=j3yBBXNct9M) (N332 Lochem live HD PTZ)
- [Camera 4](https://www.youtube.com/watch?v=keIFkcf6B5k) (N348a Den Elterweg, Zutphen live HD PTZ)
- [Camera 5](https://www.youtube.com/watch?v=Sex3fwYwQ0w) (N302 Aquaduct Harderwijk live ultraHD PTZ)
- [Camera 6](https://www.youtube.com/watch?v=Sk0aQxTygxo) (N325b Arnhem HD PTZ)

The output of the program will look as follows:

![frame](frame.jpg)


