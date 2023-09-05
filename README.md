# FingerPuppet
<!-- Ref:  https://github.com/othneildrew/Best-README-Template-->
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->



<!-- PROJECT LOGO -->
<br />
<!-- Logo comes from: https://thenounproject.com/icon/finger-walk-434686/ -->
<div align="center">
  <a href="https://thenounproject.com/icon/finger-walk-434686/">
    <img src="https://raw.githubusercontent.com/Louislar/MotionStyleTransferInHand/release/noun-finger-walk-434686.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">FingerPuppet: finger-walking performance-based puppetry for human avatar</h3>

  <p align="center">
<!--     project_description -->
      In the article we proposed a motion retargeting method, which is capable to retarget finger-walking motion to motion of human avatar. The research article has been accepted by ACM TAICHI 2023.
    <br />
    <a href="http://graphics.im.ntu.edu.tw/~robin/docs/taichi23_liang.pdf"><strong>[Paper(pdf)]</strong></a>
    <a href="http://graphics.im.ntu.edu.tw/~robin/docs/taichi23_liang.pdf"><strong>[Slide(pdf)]</strong></a>
    <br />
    <br />
    <a href="https://youtu.be/Fn-zMJrze_Y">View Demo (Youtube link)</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
<!--     <li><a href="#contributing">Contributing</a></li> -->
    <li><a href="#contact">Contact</a></li>
    <li><a href="#copyright">Copyright</a></li>
<!--     <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<!-- 講一下這個project該如何使用 -->
<!-- 補一下實機畫面, 這裡給Unity的運作畫面, 從demo影片當中截圖即可 -->
<!-- 上面project logo給的圖片用逐步visualize的圖片就好 -->

[![Application Screen Shot][application-screenshot]](https://github.com/Louislar/MotionStyleTransferInHand)   
      
This project used MediaPipe to achieve hand pose estimation, then a two-step retargeting method is utilized for generating finger motion to a 17 bones human skeletal motion. The retargeting method consists of **Lower body motion retargeting** and **Full-body pose reconstrcution**.   

A HTTP server will be established and constantly output the retargeted skeltal data stream. The skeletal data stream is stored are send in JSON format. A detail explanation is written in [Skeletal data format](#Skeletal-data-format).

This repo stores the essential code that can reproduce the retargeting result in the article. <strong> But not including the code that visualizes the resulting skeletal motion.</strong>   

[Another repository](https://github.com/Louislar/HandPuppetryUnityProjectBackup) is necessary to reproduce the Storytelling application prototype. That repo includes a Unity project which receive the retargeted skeletal poses via Http protocal, and visualize the skeletal in a 3D scene.

### Built With
[![MediaPipe][MediaPipe.logo]][MediaPipe-url]
[![Scikit-learn][Scikit-learn.logo]][Scikit-learn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

<!-- This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps. -->

A computer with Windows OS is required (not tested if linux or mac will work).  
Python interpreter must installed, and the verified version is 3.8.12. 

### Prerequisites

* Python: 3.8
* MediaPipe: 0.8.9
* scikit-learn: 1.0.2
* scipy: 1.7.3

<!-- This is an example of how to list things you need to use the software and how to install them. -->
<!-- * npm
  ```sh
  npm install npm@latest -g
  ``` -->

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Louislar/MotionStyleTransferInHand.git
   ```
2. Check the server's ip and port in the ```realTimeTestingStage.py```, line 559.
4. Also check the input source is webcam or a pre-recorded finger-walking video at line 566 in the same python script```realTimeTestingStage.py```  
Pass `0` as the first parameter to `captureByMediaPipe()`, when using a wecam.  
Pass a string of a file's address, when using a prerecorded video.  
Passing other integer is possible for using other webcam-like devices. Please refer to the parameter of `cv2.VideoCapture()`
6. Execute the following command
    ```sh
    python realTimeTestingStage.py
    ```
The retargeted body skeleton results will be printed in the terminal. Also, a server is ready for handling HTTP requests.   

Also note that the webcam tested output video in resolution of 848x480.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

After finishing all the steps in [Installation](#Installation), you can simply using HTTP GET requests to aquire retargeted human avatar's pose in a JSON format. (format is explained in [Skeletal data format](#Skeletal-data-format))

Using a HTTP GET with `/{action id}` will change the target action to the action correspond to `{action id}`. All the action ids are listed below. And it is defined at line 506 to 512 in `realTimeTestingStage.py`.

| action id  |
| ---------- |
| frontKick  |
| sideKick   |
| runSprint  |
| jumpJoy    |
| twoLegJump |


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Skeletal data format -->
## Skeletal data format
After execute the code ```realTimeTestingStage.py``` a HTTP server is established and able to handle GET request for sending the reategeted pose in body joints.   
<!-- todo: 說明body skeleton的格式 -->
The body skeletal data consists of 17 joints and can form 16 bones. Each bone is consist of two joints. All the joints and bones are listed below.   

<details>
    <summary>Table of body joints</summary>
    

| Joint index | Joint name     |
| ----------- | -------------- |
| 0           | LeftUpperLeg |
| 1           | LeftLowerLeg   |
| 2           | LeftFoot       |
| 3           | RightUpperLeg  |
| 4           | RightLowerLeg  |
| 5           | RightFoot      |
| 6           | Hip            |
| 7           | Spine          |
| 8           | Chest          |
| 9           | UpperChest     |
| 10          | LeftUpperArm   |
| 11          | LeftLowerArm   |
| 12          | LeftHand       |
| 13          | RightUpperArm  |
| 14          | RightLowerArm  |
| 15          | RightHand      |
| 16          | Head           |

</details>
<details>
    <summary>Table of body bones</summary>
    
| Bone name     | Joint pair (index) |
| ------------- | ------------------ |
| LeftHip       | 6,0                |
| RightHip      | 6,3                |
| LeftUpperLeg  | 0,1                |
| LeftLowerLeg  | 1,2                |
| RightUpperLeg | 3,4                |
| RightLowerLeg | 4,5                |
| Spine         | 6,7                |
| Chest         | 7,8                |
| UpperChest    | 8,9                |
| LeftShoulder  | 9,10               |
| LeftArm       | 10,11              |
| LeftForeArm   | 11,12              |
| RightShoulder | 9,13               |
| RightArm      | 13,14              |
| RightForeArm  | 14,15              |
| Neck          | 9,16               |

</details>



<!-- ROADMAP -->
## Roadmap

- [x] Write a clear Readme for reproduction
    - [x] Still missing the desciption of the body skeletal data format (in JSON)
- [ ] Reorganize the code base
    - [ ] Delete unnecessary codes
- [ ] Correct the Slide's hyperlink

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
## Contact

CH Liang - [EmailMe](mailto:r09922a02@cmlab.csie.ntu.edu.tw)

Project Link: [https://github.com/Louislar/MotionStyleTransferInHand](https://github.com/Louislar/MotionStyleTransferInHand)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- FIGURE -->
## Copyright

The finger-walking icon at the begining is created by Sergey Demushkin from Noun Project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[application-screenshot]:https://raw.githubusercontent.com/Louislar/MotionStyleTransferInHand/release/FingerPuppetTwoStepVis.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[UnityProject-url]:https://github.com/Louislar/HandPuppetryUnityProjectBackup
[Scikit-learn.logo]:https://img.shields.io/badge/scikit--learn-faf5f0?style=for-the-badge&logo=scikitlearn
[Scikit-learn-url]:https://scikit-learn.org/
[MediaPipe-url]:https://developers.google.com/mediapipe
[MediaPipe.logo]:https://img.shields.io/badge/MediaPipe-3ee6c1?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI%2FPgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDIwMDEwOTA0Ly9FTiIKICJodHRwOi8vd3d3LnczLm9yZy9UUi8yMDAxL1JFQy1TVkctMjAwMTA5MDQvRFREL3N2ZzEwLmR0ZCI%2BCjxzdmcgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiB3aWR0aD0iMjQwLjAwMDAwMHB0IiBoZWlnaHQ9IjI0MC4wMDAwMDBwdCIgdmlld0JveD0iMCAwIDI0MC4wMDAwMDAgMjQwLjAwMDAwMCIKIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgoKPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsMjQwLjAwMDAwMCkgc2NhbGUoMC4xMDAwMDAsLTAuMTAwMDAwKSIKZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSJub25lIj4KPHBhdGggZD0iTTQ0MiAyMDcwIGMtNDggLTExIC05OCAtNjMgLTExMSAtMTE2IC0xMyAtNTIgLTE0IC00OTIgLTEgLTU0OCAxMQotNTAgNjUgLTEwNSAxMTMgLTExNiA4NSAtMTkgMTcxIDM3IDE4OCAxMjEgMTEgNTkgMTEgNDk0IDAgNTQzIC0xMCAzOSAtNjEKMTAyIC05MCAxMTAgLTQzIDExIC02OSAxMyAtOTkgNnoiLz4KPHBhdGggZD0iTTkyMiAyMDcwIGMtNDggLTExIC05OCAtNjMgLTExMSAtMTE2IC0xNCAtNTMgLTE1IC05NzEgLTEgLTEwMjggMTEKLTUwIDY1IC0xMDUgMTEzIC0xMTYgODUgLTE5IDE3MSAzNyAxODggMTIxIDExIDU5IDExIDk3NCAwIDEwMjMgLTEwIDM5IC02MQoxMDIgLTkwIDExMCAtNDMgMTEgLTY5IDEzIC05OSA2eiIvPgo8cGF0aCBkPSJNMTQwMSAyMDcwIGMtMTggLTUgLTQ5IC0yMyAtNjggLTQyIC02MiAtNjIgLTYyIC0xNTMgMCAtMjE1IDEwOQotMTEwIDI4OCAtOCAyNTYgMTQ1IC03IDM1IC02MSA5OSAtODggMTA2IC00MyAxMSAtNjkgMTMgLTEwMCA2eiIvPgo8cGF0aCBkPSJNMTg4MiAyMDcwIGMtNDggLTExIC05OCAtNjMgLTExMSAtMTE2IC0xNCAtNTMgLTE1IC0xNDUwIC0xIC0xNTA4CjExIC01MCA2NSAtMTA1IDExMyAtMTE2IDg1IC0xOSAxNzEgMzcgMTg4IDEyMSAxMSA1OSAxMSAxNDU0IDAgMTUwMyAtMTAgMzkKLTYxIDEwMiAtOTAgMTEwIC00MyAxMSAtNjkgMTMgLTk5IDZ6Ii8%2BCjxwYXRoIGQ9Ik0xNDAyIDE1OTAgYy00OCAtMTEgLTk4IC02MyAtMTExIC0xMTYgLTE0IC01MyAtMTUgLTk3MSAtMSAtMTAyOCAxMQotNTAgNjUgLTEwNSAxMTMgLTExNiA4NSAtMTkgMTcxIDM3IDE4OCAxMjEgMTEgNTkgMTEgOTc0IDAgMTAyMyAtMTAgMzkgLTYxCjEwMiAtOTAgMTEwIC00MyAxMSAtNjkgMTMgLTk5IDZ6Ii8%2BCjxwYXRoIGQ9Ik00NDIgMTExMCBjLTQ4IC0xMSAtOTggLTYzIC0xMTEgLTExNiAtMTMgLTUyIC0xNCAtNDkyIC0xIC01NDggMTEKLTUwIDY1IC0xMDUgMTEzIC0xMTYgODUgLTE5IDE3MSAzNyAxODggMTIxIDExIDU5IDExIDQ5NCAwIDU0MyAtMTAgMzkgLTYxCjEwMiAtOTAgMTEwIC00MyAxMSAtNjkgMTMgLTk5IDZ6Ii8%2BCjxwYXRoIGQ9Ik05MjEgNjMwIGMtMTggLTUgLTQ5IC0yMyAtNjggLTQyIC02MiAtNjIgLTYyIC0xNTMgMCAtMjE1IDEwOSAtMTEwCjI4OCAtOCAyNTYgMTQ1IC03IDM1IC02MSA5OSAtODggMTA2IC00MyAxMSAtNjkgMTMgLTEwMCA2eiIvPgo8L2c%2BCjwvc3ZnPgo%3D

