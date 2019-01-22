This folder contains scripts to reproduce the figures appearing [the companion paper](https://www.biorxiv.org/content/early/2018/06/05/339564).

```
@article{giovannucci2018caiman,
  title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={bioRxiv},
  pages={339564},
  year={2018},
  publisher={Cold Spring Harbor Laboratory}
}
```


The scripts `figure_4*, figure_5*, figure_6*` load pre-saved results files for CaImAn batch and CaIman online and reproduce the
corresponding figures. These files are `all_res_web.npz` for CaImAn batch and `all_res_online_web_bk.npz` for CaImAn online,
respectively. To run the algorithms and generate these files you can execute the scripts `./preprocessing_files/Preprocess_CaImAn_batch.py/`
for CaImAn batch and `./preprocessing_files/Preprocess_CaImAn_online.py/` for CaImAn online, respectively. All files are available on Zenodo. 


### INSTRUCTIONS

Download and install CaImAn

In order to get annotations and original movies: The files can be downloaded from [zenodo](https://zenodo.org/record/1659149#.XDX8T89Ki9s) or from our [internal website](https://users.flatironinstitute.org/~neuro/caiman_paper/). BEWARE: downloading all the datasets will take approximately 400GB of space(!), but you can choose to download only a smaller subset of the datasets.

In order to simply get the annotations from the movies:
- Download and unzip WEBSITE_basic.zip
- browse the  files in the "regions" subfolder for each dataset, you will find L1_regions.json, L2_regions.json, .. etc. Where L1,L2, L3 and L4 are the annotations from different labelers. The json format is the same as the one used in neurofinder.   
Notice that these are the raw labels from the annotators. 
- The consensus among annotators is in the file consensus_regions.json. Consider that before using these labels, you might want to remove some duplicates (the labelers sometimes added the same neuron two or three times, it does happen rarely though). 

In order to reproduce results and get annotations and movies:
- Download and unzip WEBSITE.zip (this file contains also all the raw annotations contained in WEBSITE_basic.zip, so you only need to download this file to access the original annotations)
- Download all the images_XXX.zip files into the XXX/images/ folder renaming to "images.zip" (notice the XXX stands for the name of the folder . YST, N.01.01, ... etc).
Example save images_YST.zip as the file WEBSITE/YST/images/images.zip
- in rhe CaImAn repository file  CaImAn/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_batch.py  change line
base_folder = ... 
into
base_folder = '/path_to/WEBSITE' where path_to is the path where you unzipped the WEBSITE.zip folder
This step should be performed in every file used whenever the variable base_folder is present!!
- Run file CaImAn repository file  CaImAn/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_batch.py
- Same can be done for CaImAn/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_online.py

With these you can regenerate the results presented in the paper.


