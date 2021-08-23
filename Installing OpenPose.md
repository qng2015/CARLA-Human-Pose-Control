Installing OpenPose from source was unsuccessful for me. Therefore, I went with the Portable version. There are different prerequisites for different OS, all listed here: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/1_prerequisites.md#windows-prerequisites
Next, download the latest version of OpenPose, pick the one appropriate with your hardware (cpu or gpu): https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
![image](https://user-images.githubusercontent.com/60516143/130527932-40bff783-49a0-4232-b91d-6bae6abb9008.png)

Extract the zip folder and then to test run OpenPose, use PowerShell, navigate to the root of the newly extracted folder and type: 

    bin\OpenPoseDemo.exe --video examples/media/video.avi
If this doesn’t work or return an error about CUDA out of memory, try lower the resolution:

    bin\OpenPoseDemo.exe --video examples/media/video.avi --net_resolution -1x256
This number 256 in this command can be replaced with a lower number but must be a multiple of 16.
