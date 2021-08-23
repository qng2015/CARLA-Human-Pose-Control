# CARLA-Human-Pose-Control
Carla is an open source autonomous driving simulation that utilizes Unreal Engine. The user control is done through Python and C++ API (I used Python). However, what if I want to direct CARLA by using human pose. The GIF animation below shows the desired result.

![carla demo 2](https://user-images.githubusercontent.com/60516143/130524752-5bbe301a-4c64-4288-aa7c-51baf4955a6a.gif)

In this report, I will explain in detail the steps needed to install and run Carla and my python program in **windows operating system**
From this link, click on “Quick Start”: https://carla.readthedocs.io/en/0.9.10/
 ![image](https://user-images.githubusercontent.com/60516143/130523697-9e38539b-7cc0-43e9-a527-f7edc5236f55.png)
 
Notice the version number at the bottom right corner. Mine was 0.9.10. That may be different by the time you read this. You can click on the bottom right box to select any prior version you like. 
	Next screen after pressing “Quick Start” :
![image](https://user-images.githubusercontent.com/60516143/130523866-9e9b05d2-ea64-4200-8776-92a62cbb9974.png)
On this screen, check the requirements. Since I plan to run it on Windows, I only needed to make sure that I had enough hard drive space and Python installed along with the Pygame module. More on Python later. Scroll down to “B. Package installation” and click on CARLA Repository button:
![image](https://user-images.githubusercontent.com/60516143/130523892-6912c534-1669-4ef6-a516-ce9b8e38ced7.png)
This will take you to Carla Github. There are many options here. Pick the version that you want. Again, mine was 0.9.10
![image](https://user-images.githubusercontent.com/60516143/130523926-a1b56daf-8181-4d3a-ab35-973f92d8531c.png)

This download may take a while. So in the meantime, let’s move on to installing Python. Each Carla version may use a different version of Python. Carla doesn’t tell exactly what version of Python it used on their website. And just because we have the latest Python doesn’t guarantee Carla API will run. Carla 0.9.10 used Python 3.7 while I had Python 3.8 installed at the time, and the Python script I wrote did not work for me. So to get around this, I recommend installing Anaconda. Download link: https://docs.anaconda.com/anaconda/install/windows/
