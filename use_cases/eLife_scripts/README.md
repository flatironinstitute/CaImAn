This folder contains scripts to reproduce the figures appearing [the companion paper](https://elifesciences.org/articles/38173).

```
@article{giovannucci2019caiman,
  title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={eLife},
  volume={8},
  pages={e38173},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}
```


The scripts `figure_4*, figure_5*, figure_6*` load pre-saved results files for CaImAn batch and CaIman online and reproduce the
corresponding figures. These files are `all_res_web.npz` for CaImAn batch and `all_res_online_web_bk.npz` for CaImAn online,
respectively. To run the algorithms and generate these files you can execute the scripts `./preprocessing_files/Preprocess_CaImAn_batch.py/`
for CaImAn batch and `./preprocessing_files/Preprocess_CaImAn_online.py/` for CaImAn online, respectively. All files are available on Zenodo. 


### Instructions

Download and install CaImAn

In order to get annotations and original movies: The files can be downloaded from [zenodo](https://zenodo.org/record/1659149#.XDX8T89Ki9s) or from our [internal website](https://users.flatironinstitute.org/~neuro/caiman_paper/). BEWARE: downloading all the datasets will take approximately 400GB of space(!), but you can choose to download only a smaller subset of the datasets.

To simply get the raw annotations from each labeller (and their consensus):
- Download and unzip WEBSITE_basic.zip
- Browse the files in the "regions" subfolder for each dataset, you will find L1_regions.json, L2_regions.json, .. etc. Where L1,L2, L3 and L4 are the annotations from different labelers. The json format is the same as the one used in neurofinder.   
Notice that these are the raw labels (in the form of binary masks) from each annotator. 
- The consensus among annotators is in the file consensus_regions.json. Consider that before using these labels, you might want to remove some duplicates (the labelers sometimes added the same neuron two or three times, it does happen rarely though).

In order to reproduce results and get annotations and movies:
- Download and unzip WEBSITE.zip (this file contains also all the raw annotations contained in WEBSITE_basic.zip, so you only need to download this file to access the original annotations)
- Download all the images_XXX.zip files into the XXX/images/ folder renaming to "images.zip" (notice the XXX stands for the name of the folder . YST, N.01.01, ... etc).
Example save images_YST.zip as the file WEBSITE/YST/images/images.zip
- in the CaImAn repository file  [CaImAn/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_batch.py](https://github.com/flatironinstitute/CaImAn/blob/master/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_batch.py)  change line
base_folder = ... 
into
base_folder = '/path_to/WEBSITE' where path_to is the path where you unzipped the WEBSITE.zip folder
This step should be performed in every file used whenever the variable base_folder is present!!
- Run file CaImAn repository file  [CaImAn/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_batch.py](https://github.com/flatironinstitute/CaImAn/blob/master/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_batch.py)
- Same can be done for [CaImAn/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_online.py](https://github.com/flatironinstitute/CaImAn/blob/master/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_online.py)

With these you can regenerate the results presented in the paper.

### A note on reproducibility

As explained in the companion paper [(Figure 3 - figure_supplement1)](https://elifesciences.org/articles/38173/figures#fig3s1) the binary masks cannot be readily used to measure performance. We performed the following pre-processing steps:
- We constructed real valued spatial footprints as explained in the paper.
- Screened for very small components (presumably mistaken as neurons from the annotators)
- Screened for (obvious) duplicates. 
The number of removed components was always very low (no more than 5 cells) and the scripts used for this are included in our pre-processing scripts. See for example [lines 302-305 in the script for CaImAn online](https://github.com/flatironinstitute/CaImAn/blob/master/use_cases/eLife_scripts/preprocessing_files/Preprocess_CaImAn_online.py#L302)
