# Modeling User Retention through Generative Flow Networks

🐧 Implementation of GFN4Retention on KuaiSim for Kuairand-Pure Dataset

![Overview of GFN4Retention Framework](https://github.com/LZR-S/GFN4Retention/blob/main/GFN4Ret.png)

# 0.Setup

```
conda create -n KRL python=3.8
conda activate KRL
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn tqdm ipykernel
python -m ipykernel install --user --name KRL --display-name "KRL"
```

# 1. Simulator Setup

### Data Processing

See preprocess/KuaiRandDataset.ipynb for details


## 1.1 User Model

1.1 Immediate User Response Model

Example raw data format in preprocessed KuaiRand: 

> (session_id, request_id, user_id, video_id, date, time, is_click, is_like, is_comment, is_forward, is_follow, is_hate, long_view)

Example item meta data format in preprocessed KuaiRand: 

> (video_id, video_type, upload_type, music_type, log_duration, tag)

Example user meta data format in preprocessed KuaiRand: 

> (user_active_degree, is_live_streamer, is_video_author, follow_user_num_range, fans_user_num_range, friend_user_num_range, register_days_range, onehot_feat{0,1,6,9,10,11,12,13,14,15,16,17})

```
bash train_multi_behavior_user_response.sh
```

Note: multi-behavior user response models consists the state_encoder that is assumed to be the ground truth user state transition model.

1.2 User Retention Model

Pick a multi-behavior user response model for cross-session generation and retention model training, change the shell script accordingly (by setting the keyword 'KRMB_MODEL_KEY').

Generate user retention data in format:

> (session_id, user_id, session_enc, return_day)

```
bash generate_session_data.sh
```

# 2 Retention Optimization

User retention happens after leaving the previous session and identifies the beginning of the next session.

## 2.1 Setup

Evaluation metrics and protocol

**Return time** is the average time gap between the last request of the session and the first request of the session. 

**User retention** is the average ratio of visiting the system again.

## 2.2 Training for GFN model

Full model:

```
bash train_gfn_wif_kpure_cuminf_PAUR_crosssession.sh
```

Ablation without immediate feedback:

```
bash train_gfn_kpure_cuminf_PAUR_crosssession.sh
```


# 3. Result Observation

Training curves check:

> TrainingObservation.ipynb
