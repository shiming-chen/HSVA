# HSVA-NeurIPS-21
This is the implementation of “HSVA: Hierarchical Semantic-Visual Adaptation for Zero-Shot Learning” in Pytorch. The work is anonymously submitted to NeurIPS'21.
Note that this repository includes the trained model and test scripts, which used for testing and check our results repoted in our paper. Once our paper accepted, we will release the total codes of this work.

## Preparation
1. Datasets can be downloaded [Here](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).  Put it in the `data` dir.
2. Download our pretrained models from google drive [Here](https://drive.google.com/drive/folders/1h_hX114jLEa2ah5k1_Yp1nPoclinuRCw?usp=sharing), including CUB, SUN, AWA1 and AWA2 models. Put it in the `result` dir.

## Test
To test the results for GZSL or CZSL, for example:
```
CUDA_VISIBLE_DEVICES="2" python test.py --dataset CUB  --generalized True
```
`--gdataset` test dataset, i.e., CUB, SUN, AWA1, AWA2.

`--generalized` test for GZSL or CZSL.
 
