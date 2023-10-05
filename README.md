# ViFiT

Repository of our paper accepted in [MobiCom 2023](https://sigmobile.org/mobicom/2023/program.html) [ISACom](http://www.isac-2022.org/) Workshop:

**Bryan Bo Cao**, Abrar Alali, Hansi Liu, Nicholas Meegan, Marco Gruteser, Kristin Dana, Ashwin Ashok, Shubham Jain, **ViFiT: Reconstructing Vision Trajectories from IMU and Wi-Fi Fine Time Measurements**, 2023 The 29th Annual International Conference On
Mobile Computing And Networking (MobiCom), 3rd ACM MobiCom Workshop on Integrated Sensing and Communication Systems for IoT (ISACom).

ISACom '23: Proceedings of the 3rd ACM MobiCom Workshop on Integrated Sensing and Communications SystemsOctober 2023
Pages 13–18 https://doi.org/10.1145/3615984.3616503

## Abstract
Tracking subjects in videos is one of the most widely used functions in camera-based IoT applications such as security surveillance, smart city traffic safety enhancement, vehicle to pedestrian communication and so on. In the computer vision domain, tracking is usually achieved by first detecting subjects, then associating detected bounding boxes across video frames. Typically, frames are transmitted to a remote site for processing, incurring high latency and network costs. To address this, we propose ViFiT, a transformer-based model that reconstructs vision bounding box trajectories from phone data (IMU and Fine Time Measurements). It leverages a transformer's ability of better modeling long-term time series data. ViFiT is evaluated on Vi-Fi Dataset, a large-scale multimodal dataset in 5 diverse real-world scenes, including indoor and outdoor environments. Results demonstrate that ViFiT outperforms the state-of-the-art approach for cross-modal reconstruction in LSTM Encoder-Decoder architecture X-Translator and achieves a high frame reduction rate as 97.76% with IMU and Wi-Fi data.

## Motivation
Learning lightweight phone sensor data with rich motion information by a transformer model to reconstruct trajectories in long missing frames, which reduce the volume of data transmitted via network.
![Screenshot from 2023-10-05 15-42-31](https://github.com/bryanbocao/vifit/assets/14010288/ee49af25-cd1e-49ec-b792-0822a38e065d)

## Dataset
### Dataset for Model - dfv4.2
Download ```RAN4model_dfv4.2``` from [Google Drive](https://drive.google.com/drive/folders/17w8c8KK8hx1NDrjNihgNV_3Y1tdw1cJH?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AqkVlEZgdjnYcNFZsfjWUTbs12o?e=U3oRfW) and follow the folder structure:
```
ViFiT
  |-Data
     |-checkpoints
     |-datasets
        |-RAN4model_dfv4.2
  |-src
     |-...
```


### Vi-Fi Dataset
[Original Dataset link](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home), [paperswithcode link](https://paperswithcode.com/dataset/vi-fi-multi-modal-dataset).

## Pre-trained Models
Pre-trained models trained by DIoU loss can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1BLqqK6U6l3oJnIpUAsGztQ3a7oAyqzdH?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AqkVlEZgdjnYlGhbOzrpskp6TLTz?e=vPcZBl).

## Code Instructions
It is recommended to use Docker in this work. I have provided the environments in the [bryanbocao/vifit](https://hub.docker.com/repository/docker/bryanbocao/vifit/general]) container.
Usage:
#### Pulling Docker Image
```
docker pull bryanbocao/vifit
```
#### Check Images
```
docker image ls
```
```
REPOSITORY                                                               TAG       IMAGE ID       CREATED         SIZE
bryanbocao/vifit                                                         latest    6c4d67f3d122   2 months ago    13.5GB
```

#### Run and Detach a Container
```
docker run -d --ipc=host --shm-size=16384m -it -v /:/share --gpus all --network=bridge bryanbocao/vifit /bin/bash
```

#### Show Running Containers
```
docker ps -a
```
```
CONTAINER ID   IMAGE                                                                           COMMAND       CREATED        STATUS                    PORTS     NAMES
489616f0a862   bryanbocao/vifit                                                                "/bin/bash"   5 days ago     Up 5 days                           cranky_haibt
```
#### Enter a Container
```
docker exec -it <CONTAINER_ID> /bin/bash
```
In this example:
```
docker exec -it 489616f0a862 /bin/bash
```

# Citation
BibTeX:
```
@inproceedings{cao2023vifit,
  title={ViFiT: Reconstructing Vision Trajectories from IMU and Wi-Fi Fine Time Measurements},
  author={Cao, Bryan Bo and Alali, Abrar and Liu, Hansi and Meegan, Nicholas and Gruteser, Marco and Dana, Kristin and Ashok, Ashwin and Jain, Shubham},
  booktitle={Proceedings of the 3rd ACM MobiCom Workshop on Integrated Sensing and Communications Systems},
  pages={13--18},
  year={2023}
}
```
