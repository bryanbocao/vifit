# Data

#### News:
01/16/2024 We released the synchronized version (**RAN4model_dfv4p4**) of our data for future usage. This version is convenient for your research without undergoing preprocessing the raw data again. You may download the data in this [Google Drive link](https://drive.google.com/drive/folders/1fZmWWQoNIhkd7Fk9HhziCPotaCppHOLt?usp=sharing) or [OneDrive link](https://1drv.ms/f/s!AqkVlEZgdjnYoG6k1wMfgyrvbmSU?e=SRbar2).

**RAN4model_dfv4p4** provides you with the convenient synchronized format for downstream tasks. In this document, we take one subject in scene4 from one outdoor sequence as an example to demonstrate the format.

## Folder Structure
```
RAN4model_dfv4p4
  |-exps
    |-exp1
      |-start_end_ts16_dfv4p4_indoor.json
      |-start_end_ts16_dfv4p4_outdoor.json
  |-seqs
    |-indoor
      |-scene0
        |-<seq_id>
        |-20201223_140951
        |-20201223_141456
        |-20201223_141923
          |-RGBg_ts16_dfv4p4_ls.json
          |-RGBh_ts16_dfv4p4_ls.json
            |-sync_ts16_dfv4p4
              |-BBX5H_sync_dfv4p4.pkl
              |-BBXC3H_sync_dfv4p4.pkl
              |-FTM_sync_dfv4p4.pkl
              |-FTM_li_sync_dfv4p4.pkl
              |-IMU19_sync_dfv4p4.pkl
              |-IMUagm9_sync_dfv4p4.pkl
              |-RSSI_sync_dfv4p4.pkl
              |-RSSI_li_sync_dfv4p4.pkl
              ...
        ...
    |-outdoor
      |-scene1
      |-scene2
      |-scene3
      |-scene4
        |-20211007_135415
        |-20211007_135800
        |-20211007_135010
          |-RGBg_ts16_dfv4p4_ls.json
          |-RGB_ts16_dfv4p4_ls.json
          |-sync_ts16_dfv4p4
            |-BBX5_sync_dfv4p4.pkl
            |-BBX5_Others_sync_dfv4p4.pkl
            |-BBXC3_sync_dfv4p4.pkl
            |-BBXC3_Others_sync_dfv4p4.pkl
            |-FTM_sync_dfv4p4.pkl
            |-FTM_li_sync_dfv4p4.pkl
            |-IMU19_sync_dfv4p4.pkl
            |-IMUagm9_sync_dfv4p4.pkl
            |-RSSI_sync_dfv4p4.pkl
            |-RSSI_li_sync_dfv4p4.pkl
            |-Others_id_ls.pkl
       ...
```

## Data Loading
```
import pickle as pkl
with open(<M_sync_dfv4p4_path>, 'rb') as f:
    M_sync_dfv4p4 = pkl.load(f); print(f'{M_sync_dfv4p4_path} loaded!')
```
where ```M``` stands for ```modality```. You may replace ```M``` with any modality such as ```BBX5_sync_dfv4p4```.

When loading the json file, in case of the ```not JSON serializable``` error, add the following lines of code:
```
# TypeError: Object of <some_code> is not JSON serializable
# Ref: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
```

## Data Format
The shape of one synchronized file follows this format:
```
(<WIN_IDX>, <SUBJ_IDX>, <DIM_PER_FRAME>, <FEAT_AXIS_IDX>)
```
where ```<WIN_IDX>``` is window index; ```<SUBJ_IDX>``` is subject index; ```<DIM_PER_FRAME>``` is number of dimensions in one frame, it is generally set to be 1 in this version, only used for data with sampling rate higher than the current synchronized data like denser IMU data; ```<FEAT_AXIS_IDX>``` is feature axis index.

Below is an example of sequence ```20211007_144525``` in ```scene4```:
```
np.shape(BBXC3_sync_dfv4p4):  (1831, 3, 1, 3)
np.shape(BBX5_sync_dfv4p4):  (1831, 3, 1, 5)
np.shape(IMU19_sync_dfv4p4):  (1831, 3, 1, 19)
np.shape(IMUlgq10_sync_dfv4p4):  (1831, 3, 1, 10)
np.shape(IMUlgqm13_sync_dfv4p4):  (1831, 3, 1, 13)
np.shape(IMUagm9_sync_dfv4p4):  (1831, 3, 1, 9)
np.shape(FTM_sync_dfv4p4):  (1831, 3, 1, 2)
np.shape(FTM_li_sync_dfv4p4):  (1831, 3, 1, 2)
np.shape(RSSI_sync_dfv4p4):  (1831, 3, 1, 1)
np.shape(RSSI_li_sync_dfv4p4):  (1831, 3, 1, 1)
len(RGB_ts16_dfv4p4_ls):  2001
```
The synchronized data treats RGB frame as anchor where all frames named after timestamps (ts16) are saved in ```RGB_ts16_dfv4p4_ls```. In this example, sequence ```20211007_144525``` has ```2001``` frames.

## Visualization
Below are some visualizations of the data for ```subj0``` in sequence ```20211007_144525```. Note that by default, missing data points are filled by aligning nearest timestamps in the anchor (RGB) frame. Another way is by _linear interpolation_ denoted by ```li```, which is used to fill the missing data points of a modality (FTM or RSSI in 3FPS compared to RGB in 10FPS) by linearly interpolation between the borders of two consecutive logged data points.

All visualizations can be found in these links: [Google Drive](https://drive.google.com/drive/folders/1yVCX9zdPgZTb8k9VQmnJcji2qQ9zFdxr?usp=sharing), [OneDrive](https://1drv.ms/f/s!AqkVlEZgdjnYqz5NBqd7XWpK1qq7?e=Ek2SD0).

### BBX5
<img width="1040" alt="20211007_144525_BBX5" src="https://github.com/bryanbocao/vitag/assets/14010288/b209024a-6fb8-42e4-a741-3dae63df020d">

### IMU19
<img width="1066" alt="20211007_144525_IMU19" src="https://github.com/bryanbocao/vitag/assets/14010288/e99744d9-1238-4ace-b1b5-7bbce57c4610">

### IMUagm9
<img width="1062" alt="20211007_144525_IMUagm9" src="https://github.com/bryanbocao/vitag/assets/14010288/d64df45c-8fd2-4804-9fe8-ffcac9091501">

### FTM & FTM_li
<img width="1078" alt="20211007_144525_FTM_FTM_li" src="https://github.com/bryanbocao/vitag/assets/14010288/7fe52363-6908-4729-85cb-acbcb620c00e">

### RSSI & RSSI_li
<img width="1068" alt="20211007_144525_RSSI_RSSI_li" src="https://github.com/bryanbocao/vitag/assets/14010288/8ffe752e-c1a4-4619-9bbd-2b7400a69a2d">


# Vi-Fi Dataset
[Official Dataset (Raw Data) link](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home)

[paperswithcode link](https://paperswithcode.com/dataset/vi-fi-multi-modal-dataset)

# Acknowledgement
For synchronizing GPS data, please refer to [Vi-FiDatasetProcessing](https://github.com/Hai-chao-Zhang/Vi-FiDatasetProcessing). Thank [Hai-chao-Zhang](https://github.com/Hai-chao-Zhang) for contributing the scripts.

