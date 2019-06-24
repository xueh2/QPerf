# QPerf: Automatic In-line Quantitative Myocardial Perfusion Mapping

Quantitative perfusion (QPerf) is an emerging technique to directly measure blood supply to the myocardial mussel. It is demonstrating potential to be a valid clinical tool for ischemia and many non-ischemic cardiac disease. 

This repo contains the cardiac flow mapping tool we developed, in supplement to our paper submitted to [Magnetic Resonance in Medicine](https://onlinelibrary.wiley.com/journal/15222594), titled "Automatic In-line Quantitative Myocardial Perfusion Mapping: processing algorithm and implementation". This paper is currently under review.

The Blood-tissue-exchange model (BTEX [1](by Prof. James Bassingthwaighte, https://www.physiome.org/)) is used in this study for quantitative perfusion. For fully disclosure, there are many different models proposed in past two decades [2](https://www.ncbi.nlm.nih.gov/pubmed/22173205). 

BTEX model (20 version) solves two partial differential equations for myocardial blood flow (MBF, ml/min/g), permeability-surface-area product (PS, ml/min/g), blood volume (V_b, ml/g) and interstitial volume (V_isf, ml/g).

![BTEX QPerf](./images/BTEX_pixel_size_mapping.JPG "Pixel-wise BTEX flow mapping for perfusion")

To perform perfusion fitting, two inputs are needed: Cin and y. Cin is the input function for myocardium (e.g. arterial input function, AIF, measured from dual-sequence perfusion imaging) and y is the myocardial signal. Both Cin and y should be converted to Gd concentration unit (mmol/L) or have the same scale to the Gd concentration. Otherwise, the estimated flow will be off by a scaling factor.

The software is split into two function calls:

```
Matlab_gt_BTEX20_model 
==============================================================================================
Usage: Matlab_gt_BTEX20_model 
Compute BTEX20 model
---------------------------------------------------------------------
11 Input paras:
	cin                                        : AIF in [Gd], N*1, input function, in float
	tspan                                      : time to evaluate model, in seconds
	xmesh                                      : localtion to evaluate model, in cm
	Fp                                         : plasma flow, can be a vector, ml/min/g
	Vp                                         : plasma volume, can be a vector, ml/g
	PS                                         : Permeability-surface area product, can be a vector, ml/min/g
	Visf                                       : Interstitium volume, can be a vector, ml/g
	Gp                                         : scalar, ml/min/g
	Gisf                                       : scalar, ml/min/g
	Dp                                         : scalar, cm^2/sec
	Disf                                       : scalar, cm^2/sec
4 Output para:
	sol                                        : [nt nx neq], solution of BTEX20 model
	C_e                                        : Concentration of interstium
	C_p                                        : Concentration of plasma
	Q_e                                        : [Gd] residual curve
==============================================================================================
```
This function will compute BTEX look-up-table (LUT) on given parameter grid for (Fp, PS, Vp, Visf). Usually the Gd diffusion parameters are fixed, but user can choose to play with them as well.

The second function call will perform the BTEX fitting, given the input Cin and y and pre-computed LUT:

```
Matlab_gt_BTEX_fitting
==============================================================================================
Usage: Matlab_gt_BTEX_fitting 
Perform gadgetron btex fitting
---------------------------------------------------------------------
12 Input paras:
	cin                                        : N*1, input function, in double
	y                                          : N*M, response function, in double
	deltaT                                     : sample interval of cin and y, in seconds (default, 0.5s)
	max_iter_BTEX20                            
	max_func_eval_BTEX20                       
	local_search_BTEX20                        
	Q_e                                        : BTEX 20 look up table
	Fp                                         : BTEX 20 Fp array
	PS                                         : BTEX 20 PS array
	Vp                                         : BTEX 20 Vp array
	Visf                                       : BTEX 20 Visf array
	max_shift                                  : max shift range to search, in seconds
2 Output para:
	res_BTEX20                                 : results for BTEX20, [Fp, PS, Vp, Visf]
	y_BTEX20                                   : model output for BTEX20
==============================================================================================
```

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
