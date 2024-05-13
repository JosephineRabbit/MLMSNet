# Saliency-Evaluation-Toolbox


This repository contains measures for evaluating salient object detection models in python.

- [E-measure](https://arxiv.org/abs/1805.10421)   
- [S-measure](https://www.crcv.ucf.edu/papers/iccv17/1164.pdf)   
- [Weighted F-measure](https://ieeexplore.ieee.org/document/6909433)   
- Adaptive F-measure or F-measure    
- MAE    
- PR curve
- F-measure curve (different thresholds)


### Requirements
- [opencv-python](https://github.com/skvark/opencv-python)
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [tqdm](https://github.com/tqdm/tqdm)

### Usage

Simply import `calculate_measures`, set Saliency Maps and Ground Truth paths, and choose measures you would like to be calculated (measures: `'MAE', 'E-measure', 'S-measure', 'Max-F', 'Adp-F', 'Wgt-F'`). All results can be saved as numpy arrays by specifying `save` parameter.   
```python
>>> from saliency_toolbox import calculate_measures
>>> sm_dir = 'SM/'
>>> gt_dir = 'GT/'
>>> res    = calculate_measures(gt_dir, sm_dir, ['E-measure', 'S-measure', 'Wgt-F'], save=False)
>>> print(res)

{'E-measure': 0.990753, 'S-measure': 0.958684, 'Wgt-F': 0.974209}
```

To plot F-measure and Precision-Recall curves, firstly, `Max-F` should be calculated and stored (by specifying `save`). This results in storing `Precision.npy`, `Recall.npy` and `Fmeasure_all_thresholds.npy`, and then we can use them to plot curves. For example:
 ```python
>>> res       = calculate_measures(gt_dir, sm_dir, ['Max-F'], save='./')
>>> prec      = np.load('save/Precision.npy')
>>> recall    = np.load('save/Recall.npy')
>>> f_measure = np.load('save/Fmeasure_all_thresholds.npy')
>>> plt.plot(recall, prec)
>>> plt.figure()
>>> plt.plot(np.linspace(0, 1, len(f_measure)), f_measure) 
```
### Citation
```
@article{mohammadi2020cagnet,
  title={CAGNet: Content-Aware Guidance for Salient Object Detection},
  author={Mohammadi, Sina and Noori, Mehrdad and Bahri, Ali and Majelan, Sina Ghofrani and Havaei, Mohammad},
  journal={Pattern Recognition},
  pages={107303},
  year={2020},
  publisher={Elsevier}
}
```

```
@article{noori2020dfnet,
  title={DFNet: Discriminative feature extraction and integration network for salient object detection},
  author={Noori, Mehrdad and Mohammadi, Sina and Majelan, Sina Ghofrani and Bahri, Ali and Havaei, Mohammad},
  journal={Engineering Applications of Artificial Intelligence},
  volume={89},
  pages={103419},
  year={2020},
  publisher={Elsevier}
}
```


