'''
Usage:
python3 convert_BBOX_to_YOLOv5_txt.py -ud -m snow -sc 0 -tsid_idx 5
python3 convert_BBOX_to_YOLOv5_txt.py -ud -m snow -sc 1 -tsid_idx 0
python3 convert_BBOX_to_YOLOv5_txt.py -ud -m snow -sc 2 -tsid_idx 13
python3 convert_BBOX_to_YOLOv5_txt.py -ud -m snow -sc 3 -tsid_idx 8
python3 convert_BBOX_to_YOLOv5_txt.py -ud -m snow -sc 4 -tsid_idx 4
'''

import os
from config import Config
import train
from utils import Config, preprocess_config
import json
import cv2

def convert_BBOX_to_YOLOv5_txt(C: Config):
    print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - C.normalization: ', C.normalization)
    # print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - len(test_seq_in_correspondence): ', len(test_seq_in_correspondence))
    # print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[0]): ', np.shape(test_seq_in_correspondence[0]))
    # print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[1]): ', np.shape(test_seq_in_correspondence[1]))
    # print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[2]): ', np.shape(test_seq_in_correspondence[2]))
    # print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[3]): ', np.shape(test_seq_in_correspondence[3]))
    '''
    utils.py - convert_BBOX_to_YOLOv5_txt() - C.normalization:  True
    utils.py - convert_BBOX_to_YOLOv5_txt() - len(test_data_set):  3382
    utils.py - convert_BBOX_to_YOLOv5_txt() - len(test_data_set[0]):  3
    utils.py - convert_BBOX_to_YOLOv5_txt() - len(test_data_set[1]):  3
    utils.py - convert_BBOX_to_YOLOv5_txt() - len(test_data_set[2]):  3
    utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_data_set[0][0]):  (10, 1, 5)
    utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_data_set[0][1]):  (10, 1, 2)
    utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_data_set[0][2]):  (10, 1, 9)
    utils.py - convert_BBOX_to_YOLOv5_txt() - len(test_seq_in_correspondence):  4
    utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[0]):  torch.Size([2970, 10, 1, 5]) # [len_video, len_win, samples_per_win, features]
    utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[1]):  torch.Size([2970, 10, 1, 2]) # [len_video, len_win, samples_per_win, features]
    utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[2]):  torch.Size([2970, 10, 1, 19]) # [len_video, len_win, samples_per_win, features]
    utils.py - convert_BBOX_to_YOLOv5_txt() - np.shape(test_seq_in_correspondence[3]):  torch.Size([2970, 10, 1, 9]) # [len_video, len_win, samples_per_win, features]
    '''
    # seq_in_BBX5_video = test_seq_in_correspondence[0]
    # seq_in_FTM2_video = test_seq_in_correspondence[1]
    # seq_in_IMU19_video = test_seq_in_correspondence[2]
    # seq_in_IMU_XT_video = test_seq_in_correspondence[3]

    update_running_mode = True
    load_pretrained_embeds = False
    # flag = None
    trainer, _, val_data_loader, test_data_loader, \
        _, val_seq_in_correspondence, test_seq_in_correspondence, \
        criterion, optimizer, C = preprocess_config(C, 'test', update_running_mode, load_pretrained_embeds) #, flag)
    # [seq_in_BBX5, seq_in_FTM2, seq_in_IMU19]

    # print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt - C.max_depth: ', C.max_depth)
    reconstruction_path = C.reconstruction_path

    # if C.scene_id == 0: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json'
    # else: C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv.shape(BBX5_sync_dfv4)4_ls.json'
    # print('C.RGBg_ts16_dfv4_ls_path: ', C.RGBg_ts16_dfv4_ls_path)
    C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBg_ts16_dfv4_ls.json'
    print('C.RGBg_ts16_dfv4_ls_path: ', C.RGBg_ts16_dfv4_ls_path)

    with open(C.RGBg_ts16_dfv4_ls_path, 'r') as f:
        C.RGBg_ts16_dfv4_ls = json.load(f)
        print(C.RGBg_ts16_dfv4_ls_path, 'loaded!')
        print('C.RGBg_ts16_dfv4_ls[:5]: ', C.RGBg_ts16_dfv4_ls[:5]) # verification
    print('len(C.RGBg_ts16_dfv4_ls): ', len(C.RGBg_ts16_dfv4_ls))
    # e.g. 1816
    # phone_win_offset = np.shape(seq_in_IMU_XT_video)[0] - len(C.RGBg_ts16_dfv4_ls)

    

    # --------------
    #  Video Window
    # --------------
    C.crr_ts16_dfv4_ls_all_i = 0
    C.video_len = min(1700, len(C.RGBg_ts16_dfv4_ls)) # len(C.ts12_BBX5_all)
    print(); print() # debug
    print('C.video_len: ', C.video_len) # e.g. 1700
    C.n_wins = min(1700, C.video_len - C.len_win + 1)
    print('C.n_wins: ', C.n_wins) # e.g. 1791

    print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - C.seq_id_to_start_end_ts16_dfv4[C.seq_id][start]: ', C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start'])
    print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - C.seq_id_to_start_end_ts16_dfv4[C.seq_id][end]: ', C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end'])
    # print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - C.test_seq_subj_i_in_view_dict: ', C.test_seq_subj_i_in_view_dict)
    print('\n\n utils.py - convert_BBOX_to_YOLOv5_txt() - C.seq_id: ', C.seq_id)

    n_out_of_view = 0
    proc_n_frames, proc_n_wins = 0, 0 # actual processed number of frames and windows
    print('\n\n C.img_path: ', C.img_path)
    # e.g. /share/media/brcao/eData21/Data/datasets/RAN_All/RAN/seqs/outdoor/scene1/20211004_142306/RGB_ts16_dfv4_anonymized

    # YOLOv5 txt >>>
    # Load BBX file >>>
    seq_path_ = C.img_path[:C.img_path.rindex('/')]
    print('\n\n seq_path_: ', seq_path_)

    print('\n C.macro_scene: ', C.macro_scene)
    if C.macro_scene == 'indoor':
        # e.g. /share/media/brcao/eData21/Data/datasets/RAN_All/RAN/seqs/outdoor/scene1/20211004_142306/
        GND_ts16_dfv4_to_BBX5_dfv4_file_path = seq_path_ + '/GND_ts16_dfv2/GND_ts16_dfv2_to_BBX5H_dfv2.json'
        with open(GND_ts16_dfv4_to_BBX5_dfv4_file_path, 'r') as f:
            GND_ts16_dfv4_to_BBX5_dfv4 = json.load(f)
            print('\n\n ', GND_ts16_dfv4_to_BBX5_dfv4_file_path, 'loaded!')
            print('\n\n GND_ts16_dfv4_to_BBX5_dfv4: ', GND_ts16_dfv4_to_BBX5_dfv4)
    elif C.macro_scene == 'outdoor':
        # e.g. /share/media/brcao/eData21/Data/datasets/RAN_All/RAN/seqs/outdoor/scene1/20211004_142306/
        GND_ts16_dfv4_to_BBX5_dfv4_file_path = seq_path_ + '/GND_ts16_dfv4/GND_ts16_dfv4_to_BBX5_dfv4.json'
        with open(GND_ts16_dfv4_to_BBX5_dfv4_file_path, 'r') as f:
            GND_ts16_dfv4_to_BBX5_dfv4 = json.load(f)
            print('\n\n ', GND_ts16_dfv4_to_BBX5_dfv4_file_path, 'loaded!')
            print('\n\n GND_ts16_dfv4_to_BBX5_dfv4: ', GND_ts16_dfv4_to_BBX5_dfv4)
    # Load BBX file <<<

    # Save files >>>
    yolo_seq_path_ = seq_path_ + '/yolo'
    if not os.path.exists(yolo_seq_path_): os.makedirs(yolo_seq_path_); print(yolo_seq_path_, 'created!')

    train_RAN_file = open(yolo_seq_path_ + '/train_RAN.txt', 'w')
    val_RAN_file = open(yolo_seq_path_ + '/val_RAN.txt', 'w')

    images_path = yolo_seq_path_ + '/images'
    if not os.path.exists(images_path): os.makedirs(images_path); print(images_path, 'created!')
    images_val_path = images_path + '/val'
    if not os.path.exists(images_val_path): os.makedirs(images_val_path); print(images_val_path, 'created!')

    labels_path = yolo_seq_path_ + '/labels'
    if not os.path.exists(labels_path): os.makedirs(labels_path); print(labels_path, 'created!')
    labels_val_path = labels_path + '/val'
    if not os.path.exists(labels_val_path): os.makedirs(labels_val_path); print(labels_val_path, 'created!')
    # Save files <<<
    # YOLOv5 txt <<<
    
    # ------------------------
    #  Iterate Over the Video
    # ------------------------
    # for win_i in range(C.n_wins):
    for win_i in range(0, C.n_wins, C.win_stride):
        
        ts16_dfv4 = C.RGBg_ts16_dfv4_ls[win_i] # C.RGB_ts16_dfv4_valid_ls[win_i]
        if ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start']:
            # phone_win_offset += 1
            continue
        elif ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']: break
        print('\n\n win_i: ', win_i, ', ts16_dfv4: ', ts16_dfv4)
        phone_win_i = win_i # + phone_win_offset
        if ts16_dfv4 in C.test_seq_subj_i_in_view_dict.keys():
            img_path = C.img_path + '/' + ts16_dfv4 + '_anonymized.jpg'
            C.img = cv2.imread(img_path)
            print('\n\n img_path: ', img_path)
            height, width, channels = C.img.shape
            print('\n\n height, width, channels: ', height, width, channels)
            # outdoor: 720 1280 3 

            test_seq_subj_i_in_view_ls_ = C.test_seq_subj_i_in_view_dict[ts16_dfv4]
            print('\n\n test_seq_subj_i_in_view_ls_: ', test_seq_subj_i_in_view_ls_) # e.g. [1, 2, 3, 4]

            n = len(test_seq_subj_i_in_view_ls_)
            if n > 0:
                # C.ts16_dfv4_subj_i_to_BBX5_prime[ts16_dfv4] = defaultdict()
                print('\n\n C.seq_path: ', C.seq_path)
                
                # Copy txt files
                print('\n labels_val_path: ', labels_val_path)
                labels_val_file_path = labels_val_path + '/' + ts16_dfv4 + '.txt'
                # if not os.path.exists(labels_val_file_path): os.makedirs(labels_val_file_path); print(labels_val_file_path, 'created!')
                with open(labels_val_file_path, 'w') as f:
                    if ts16_dfv4 in GND_ts16_dfv4_to_BBX5_dfv4:
                        # Copy img
                        print('\n images_val_path: ', images_val_path)
                        images_val_path_ = images_val_path + '/' + ts16_dfv4 + '.jpg'
                        print('\n images_val_path_: ', images_val_path_)
                        cmd = 'cp {} {}'.format(img_path, images_val_path_); os.system(cmd); print(cmd)
                        print('\n GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4]: ', GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4])
                        for ts16_dfv4_, v in GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4].items():
                            print('\n v: ', v)
                            SAVE = False
                            if C.macro_scene == 'indoor': SAVE = True
                            elif C.macro_scene == 'outdoor':
                                # e.g. v:  {'subj': 'Hansi', 'c_row': 668, 'c_col': 724, 'w': 132, 'h': 98, 'd': 1.6318421363830566}
                                if 'Other' not in v['subj']:
                                    SAVE = True
                            if SAVE:
                                x, y = float(v['c_col']) / width, float(v['c_row']) / height
                                w, h = float(v['w']) / width, float(v['h']) / height
                                cls_id = 0
                                write_str = '{} {} {} {} {}\n'.format(cls_id, x, y, w, h)
                                f.write(write_str); print('\n', write_str, 'written!')

                                # Dummy train, train data is not used at this point actually.
                                train_write_str = './images/train/' + ts16_dfv4 + '.jpg\n'
                                train_RAN_file.write(train_write_str); print(train_write_str, 'written!')

                                val_write_str = './images/val/' + ts16_dfv4 + '.jpg\n'
                                val_RAN_file.write(val_write_str); print(val_write_str, 'written!')

if __name__ == '__main__':
    C = Config()
    convert_BBOX_to_YOLOv5_txt(C)
