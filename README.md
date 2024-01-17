# ViFiT

Repository of our paper accepted in [MobiCom 2023](https://sigmobile.org/mobicom/2023/program.html) [ISACom](http://www.isac-2022.org/) Workshop:

**Bryan Bo Cao**, Abrar Alali, Hansi Liu, Nicholas Meegan, Marco Gruteser, Kristin Dana, Ashwin Ashok, Shubham Jain, **ViFiT: Reconstructing Vision Trajectories from IMU and Wi-Fi Fine Time Measurements**, 2023 The 29th Annual International Conference On
Mobile Computing And Networking (MobiCom), 3rd ACM MobiCom Workshop on Integrated Sensing and Communication Systems for IoT (ISACom).

ISACom '23: Proceedings of the 3rd ACM MobiCom Workshop on Integrated Sensing and Communications SystemsOctober 2023
Pages 13–18 https://doi.org/10.1145/3615984.3616503

[arXiv 2310.03140](https://arxiv.org/pdf/2310.03140.pdf)

# Vi-Fi Dataset

**New** 01/16/2024: We released the synchronized version (**RAN4model_dfv4p4**) of our data for future usage. This version is convenient for your research without undergoing preprocessing the raw data again. Check out the details in the [DATA.md](https://github.com/bryanbocao/vitag/blob/main/DATA.md) file.

[Official Dataset (Raw Data) link](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home)

[paperswithcode link](https://paperswithcode.com/dataset/vi-fi-multi-modal-dataset)

## Abstract
Tracking subjects in videos is one of the most widely used functions in camera-based IoT applications such as security surveillance, smart city traffic safety enhancement, vehicle to pedestrian communication and so on. In the computer vision domain, tracking is usually achieved by first detecting subjects, then associating detected bounding boxes across video frames. Typically, frames are transmitted to a remote site for processing, incurring high latency and network costs. To address this, we propose ViFiT, a transformer-based model that reconstructs vision bounding box trajectories from phone data (IMU and Fine Time Measurements). It leverages a transformer's ability of better modeling long-term time series data. ViFiT is evaluated on Vi-Fi Dataset, a large-scale multimodal dataset in 5 diverse real-world scenes, including indoor and outdoor environments. Results demonstrate that ViFiT outperforms the state-of-the-art approach for cross-modal reconstruction in LSTM Encoder-Decoder architecture X-Translator and achieves a high frame reduction rate as 97.76% with IMU and Wi-Fi data.

## Motivation
Two types of challenges using vision-only methods: **(a) Frame Drop**, an entire frame in the next timestamp is not available (e.g. due to temporal down sampling to save network bandwidth, network losses, etc.), resulting in missing visual information for estimating object of interests’ detections (cyan); **(b) Salient Part Missing**: salient parts of objects are missing due to occlusion in the environment (purple) such as the truck or moving out of the camera’s view (orange). Missing parts are displayed in lower opacity by dotted lines. Each color represents one identity of subject of interest. Detection ground truths are shown by solid bounding boxes.
<img width="1346" alt="Screenshot 2024-01-16 at 2 18 08 PM" src="https://github.com/bryanbocao/vifit/assets/14010288/f234dcd3-797e-45e2-9b5d-0730a0849cf7">

## System Overview
Learning lightweight phone sensor data with rich motion information by a transformer model to reconstruct trajectories in long missing frames, which reduce the volume of data transmitted via network.
![Screenshot from 2023-10-05 15-42-31](https://github.com/bryanbocao/vifit/assets/14010288/ee49af25-cd1e-49ec-b792-0822a38e065d)

## Vi-Fi Transformer (ViFiT)
_ViFiT_ consists of multimodal Encoders for (_T<sub>c</sub><sup>0</sup>_, _T<sub>i</sub>_ and _T<sub>f</sub>_ ) to extract features and Vision Decoder to reconstruct the whole visual trajectory of _T<sub>c</sub>′_ for the missing frames in a window with length _WL_. Note _T<sub>c</sub><sup>0</sup>_ denotes a vision tracklet with first frame only and _H_ denotes representation dimension.
<img width="1463" alt="Screenshot 2024-01-16 at 2 20 09 PM" src="https://github.com/bryanbocao/vifit/assets/14010288/912a32e9-29b0-4b3d-870a-45fdef637655">

Vi-Fi Transformer (_ViFiT_) Architecture. _ViFiT_ is comprised of multimodal Encoders for (_T<sub>c</sub><sup>0</sup>_, _T<sub>i</sub>_ and _T<sub>f</sub>_ ) depicted on the left side in parallel displayed with various degrees of opacity, as well as a Vision Decoder on the right. Information flow starts from the bottom left corner, where each tracklet for one modality (_T<sub>c</sub><sup>0</sup>_, _T<sub>i</sub>_ or _T<sub>f</sub>_ ) is fed into its own Encoder independently, including B blocks of transformer modules with Multi-head Self-attention (MSA). In the next step, Encoders generate multimodal representations, fused by concatenation (_X<sub>c</sub>′_, _X<sub>i</sub>′_, _X<sub>f</sub>′_ ) and are fed into the Vision Decoder to output bounding boxes (_Tc′_) in missing frames.

<img width="1186" alt="Screenshot 2024-01-16 at 2 25 10 PM" src="https://github.com/bryanbocao/vifit/assets/14010288/bb250e9a-e10b-4a32-84d1-78b2facceb4a">

## Result
<img width="300" alt="Screenshot 2024-01-16 at 2 31 29 PM" src="https://github.com/bryanbocao/vifit/assets/14010288/5f15513b-1c02-49ed-ac9c-f4544e556818">

Samples of reconstructed vision tracklets _T<sub>c</sub>sub>′_ and ground truths _GT_ decorated in lighter (1st and 3rd rows) and darker colors (2nd and 4th rows), respectively (Best view in color). Indoor scene is shown in the 1st column while outdoor scenes are displayed from the 2nd to the 5th columns.
<img width="1571" alt="Screenshot 2024-01-16 at 2 32 21 PM" src="https://github.com/bryanbocao/vifit/assets/14010288/e8bac87d-1dbc-4bcf-99f9-35cc1e3f54b6">


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

#### Train ViFiT
Under the ```src/model_v4.2``` folder inside the container created by the commands above. Note that you need to specify ```<MACHINE_NAME>```.
```
python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 0 -tsid_idx 5 -lw 30 -lf DIOU
python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 1 -tsid_idx 0 -lw 30 -lf DIOU
python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 2 -tsid_idx 13 -lw 30 -lf DIOU
python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 3 -tsid_idx 8 -lw 30 -lf DIOU
python3 Xformer_IFcC2C.py -ud -n -rm train -te 500 -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 4 -tsid_idx 4 -lw 30 -lf DIOU
```

#### Test ViFiT
```
python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 0 -tsid_idx 5 -lw 30  -lf DIOU -ld_tr_eid -tr_eid 420 -ffo -mrf -w_s 29
python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 1 -tsid_idx 0 -lw 30  -lf DIOU -ld_tr_eid -tr_eid 204 -ffo -mrf -w_s 29
python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 2 -tsid_idx 13 -lw 30 -lf DIOU -ld_tr_eid -tr_eid 165 -ffo -mrf -w_s 29
python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 3 -tsid_idx 8 -lw 30 -lf DIOU -ld_tr_eid -tr_eid 204 -ffo -mrf -w_s 29
python3 Xformer_IFcC2C.py -ud -n -rm test -nan linear_interp -tr_md_id Xformer_IFcC2C -m <MACHINE_NAME> -sc 4 -tsid_idx 4 -lw 30 -lf DIOU -ld_tr_eid -tr_eid 248 -ffo -mrf -w_s 29
```

# Citation
ViFiT BibTeX:
```
@inproceedings{cao2023vifit,
  title={ViFiT: Reconstructing Vision Trajectories from IMU and Wi-Fi Fine Time Measurements},
  author={Cao, Bryan Bo and Alali, Abrar and Liu, Hansi and Meegan, Nicholas and Gruteser, Marco and Dana, Kristin and Ashok, Ashwin and Jain, Shubham},
  booktitle={Proceedings of the 3rd ACM MobiCom Workshop on Integrated Sensing and Communications Systems},
  pages={13--18},
  year={2023}
}
```

Vi-Fi (dataset) BibTex:
```
@inproceedings{liu2022vi,
  title={Vi-Fi: Associating Moving Subjects across Vision and Wireless Sensors},
  author={Liu, Hansi and Alali, Abrar and Ibrahim, Mohamed and Cao, Bryan Bo and Meegan, Nicholas and Li, Hongyu and Gruteser, Marco and Jain, Shubham and Dana, Kristin and Ashok, Ashwin and others},
  booktitle={2022 21st ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)},
  pages={208--219},
  year={2022},
  organization={IEEE}
}
```
```
@misc{vifisite,
  author        = "Hansi Liu",
  title         = "Vi-Fi Dataset",
  month         = "Dec. 05,", 
  year          = "2022 [Online]",
  url           = "https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home"
}
```

[Reality-Aware Networks Project Website](https://ashwinashok.github.io/realityawarenetworks/)

# Acknowledgement
This research has been supported by the National Science Foundation (NSF) under Grant Nos. CNS-2055520, CNS1901355, CNS-1901133. 
