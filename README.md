# EffecTri


EffecTri is a deep learning-based framework for accurate and interpretable prediction of Type III, IV, and VI secretion system effector proteins (T3SEs, T4SEs, T6SEs).

![image](https://github.com/lijingtju/EffecTri/blob/main/Figure2.png)

## Requirements
-GPU
## Installation environment
```shell
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Use AttenRNA predict your RNA sequence
```shell
python EffecTri_prediction.py --data test.csv --log_dir ./logs --batch 32 --resume ./human_model.pt
```

You can get the predict score from ```./log``` forder.
