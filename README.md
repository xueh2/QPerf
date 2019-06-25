# QPerf: Automatic In-line Quantitative Myocardial Perfusion Mapping

Quantitative perfusion (QPerf) is an emerging technique to directly measure blood supply to the myocardial mussel. It is demonstrating potential to be a valid clinical tool for ischemia and many non-ischemic cardiac disease. 

This repo contains the cardiac flow mapping tool we developed, in supplement to our paper submitted to [Magnetic Resonance in Medicine](https://onlinelibrary.wiley.com/journal/15222594), titled "Automatic In-line Quantitative Myocardial Perfusion Mapping: processing algorithm and implementation". This paper is currently under review.

The Blood-tissue-exchange model (BTEX [1](by Prof. James Bassingthwaighte, https://www.physiome.org/)) is used in this study for quantitative perfusion. For fully disclosure, there are many different models proposed in past two decades [2](https://www.ncbi.nlm.nih.gov/pubmed/22173205). 

BTEX model (20 version) solves two partial differential equations for myocardial blood flow (MBF, ml/min/g), permeability-surface-area product (PS, ml/min/g), blood volume (V_b, ml/g) and interstitial volume (V_isf, ml/g).

![BTEX QPerf](./images/BTEX_pixel_size_mapping.JPG "Pixel-wise BTEX flow mapping for perfusion")

To perform perfusion fitting, two inputs are needed: Cin and y. Cin is the input function for myocardium (e.g. arterial input function, AIF, measured from dual-sequence perfusion imaging) and y is the myocardial signal. Both Cin and y should be converted to Gd concentration unit (mmol/L) or have the same scale to the Gd concentration. Otherwise, the estimated flow will be off by a scaling factor.

The QPerf mapping is provided as a function call:

```
Matlab_gt_QPerf_mapping
==============================================================================================
Usage: Matlab_gt_QPerf_mapping 
Perform gadgetron perfusion flow map estimation
---------------------------------------------------------------------
7 Input paras:
	cin                                   : N*1, input function, whole range aif with baseline, in float
	y                                     : RO*E1*N, response function array, in float
	y_mask                                : RO*E1, if not empty, mask for background, background is 0 and foreground is >0, in float
	foot                                  : foot index in cin
	peak                                  : peak index in cin
	deltaT                                : time tick in ms for every data point
	hematocrit                            : the hematocrit, e.g. 0.42
1 Output para:
	flow_maps                             : flow maps in ml/min/g
==============================================================================================
```
This function will compute BTEX look-up-table (LUT) and perform pixel-wise mapping.

The **examples** folder has running example file to demonstrate the usage of these commands.

To install the software, clone this repo and add **%BASE_DIR%/QPerf/software** to your matlab path, suppose the repo was cloned to **%BASE_DIR**.

Current version of software was tested in windows 10 with Matlab R2013b to R2017b.

For more comments and suggestions, please contact me at :

```
Hui Xue
National Heart Lung and Blood Institute (NHLBI)
National Institutes of Health
10 Center Drive, MSC-1061
Bethesda, MD 20892-1061
USA

Email: hui.xue@nih.gov
```
