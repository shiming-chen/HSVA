# HSVA-NeurIPS-21
This is the implementation of “**HSVA: Hierarchical Semantic-Visual Adaptation for Zero-Shot Learning**” in Pytorch. The work is anonymously submitted to NeurIPS'21.
Note that this repository includes the trained model and test scripts, which is used for testing and checking our results repoted in our paper. <b style='color:red'>Once our paper is accepted, we will release all codes of this work</b>.

## Preparation
1. Datasets can be downloaded [Here](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).  Put it in the `data` dir.
2. Download our pretrained models from google drive [Here](https://drive.google.com/drive/folders/1h_hX114jLEa2ah5k1_Yp1nPoclinuRCw?usp=sharing), including CUB, SUN, AWA1 and AWA2 models. Put it in the `result` dir. Note that we just provide one pre-trained model for every dataset.

## Test
To test the results for GZSL or CZSL, for example:
```
CUDA_VISIBLE_DEVICES="2" python test.py --dataset CUB  --generalized True
```
`--gdataset` test dataset, i.e., CUB, SUN, AWA1, and AWA2.

`--generalized` test for GZSL or CZSL.
 
## Results
Results of our released model using various evaluation protocols on four datasets. Since we use the one pre-trained model for evaluating all protocol (e.g., top-1 accuracy on seen classes (S), top-1 accuracy on unseen classes (U), harmonic mearn (H), and top-1 accuracy on unseen classes for CZSL (acc)) using various classifiers (e.g., **softmax/1-NN/ 5-NN**), the results may slightly larger/smaller than the results reported in our paper.

|Datasets | U | S| H| acc |
| ----- | ---------- | ---------- | ---------- | ---- |
| AWA1 | 61.1/69.4/69.9 |	75.2/90.2/94.0 |	67.4/78.5/80.2 | 67.7 |
| AWA1 | 57.8/66.2/66.2	| 79.3/93.4/96.6	| 66.9/77.5/78.5 |  --  |
| CUB  | 51.9/66.2/64.9	| 59.5/74.8/65.5 |	55.5/70.2/65.2 | 60.8 |
| SUN  | 49.3/68.9/66.3	| 37.1/42.2/36.4	| 42.3/52.4/47.0 | 63.8 |

