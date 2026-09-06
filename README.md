# LP2DH sample trial on DynTex++

This repository provides a single-trial, single-scale (`P = 3`) MATLAB
implementation of LP2DH on DynTex++. It is intended as a compact example of
the main LP2DH pipeline.

Run MATLAB from this folder:

```matlab
run_lp2dh_dyntexpp
```

Set `data_root` near the top of the script if DynTex++ is stored elsewhere.
The Statistics and Machine Learning Toolbox is required.

The default script directly loads `split.mat`, `W.mat`, `dictionary.mat`, and
`features.mat`. Adjacent commented lines show how to rebuild the split, train
W and the dictionary, or recompute all video features. 

The Stiefel-manifold optimization code in `third_party/wen_yin` is adapted from 
the Wen-Yin optimization implementation and is kept separately from the LP2DH code.

## Citation

If you find this code useful in your research, please cite our paper:

@ARTICLE{11643494,
  author={Ding, Ruxin and Ren, Jianfeng and Yu, Heng and Li, Jiawei and Jiang, Xudong},
  journal={IEEE Transactions on Image Processing}, 
  title={LP2DH: A Locality-Preserving Pixel-Difference Hashing Framework for Dynamic Texture Recognition}, 
  year={2026},
  volume={35},
  number={},
  pages={8762-8774},
  doi={10.1109/TIP.2026.3718418}}
