# CARLA-Human-Pose-Control
Carla is an open source autonomous driving simulation that utilizes Unreal Engine. The user control is done through Python and C++ API (I used Python). However, what if I want to direct CARLA by using human pose. The GIF animation below shows the desired result.

![carla demo 2](https://user-images.githubusercontent.com/60516143/130524752-5bbe301a-4c64-4288-aa7c-51baf4955a6a.gif)

In this report, I will explain in detail the steps needed to install and run Carla and my python program in **windows operating system**. Firstly, we need these prequisites:
1) CARLA and Anaconda Installation: https://github.com/qng2015/CARLA-Human-Pose-Control/blob/main/Installing%20CARLA%20and%20Anaconda.md
2) Pytorch Installation: https://github.com/qng2015/CARLA-Human-Pose-Control/edit/main/Installing%20PyTorch
3) OpenPose Installation: https://github.com/qng2015/CARLA-Human-Pose-Control/blob/main/Installing%20OpenPose.md

Running my python program (usepose2.py):

Not all the python files here are important. The files inside the "learning progress" folder were simply my study progress of these different API and libraries that eventually led to the finalized **usepose2.py**. AKA you don't need to run those.

Copy **usepose2.py** and the two folders empty **input_img** and **output_images** into **CARLA_0.9.10\WindowsNoEditor\PythonAPI\examples**. Make sure you have a camera or webcam working for your PC.
To run this program:

●	Launch Carla and wait for it to load up. 

●	Meanwhile, activate the anaconda environment in the command prompt if you haven't already: 

    conda activate EnvironmentName
●	Navigate to the directory: CARLA_0.9.10\WindowsNoEditor\PythonAPI\examples 

●	Once Carla finishes loading up, type into command prompt:  

    python usepose2.py
