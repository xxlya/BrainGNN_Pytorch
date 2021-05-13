# Graph Neural Network for Brain Network Analysis
 A preliminary implementation of BrainGNN


## Usage
### Setup
**pip**

See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```
**PYG**

To install pyg library, [please refer to the document](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Dataset 
**ABIDE**
We treat each fMRI as a brain graph. How to download and construct the graphs?
```
python 01-fetch_data.py
python 02-process_data.py
```

### How to run classification?
Training and testing are integrated in file `main.py`. To run
```
python 03-main.py 
```


## Citation
If you find the code and dataset useful, please cite our paper.
```latex
@article{li2020braingnn,
  title={Braingnn: Interpretable brain graph neural network for fmri analysis},
  author={Li, Xiaoxiao and Zhou,Yuan and Dvornek, Nicha and Zhang, Muhan and Gao, Siyuan and Zhuang, Juntang and Scheinost, Dustin and Staib, Lawrence and Ventola, Pamela and Duncan, James},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```
