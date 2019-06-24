# QPerf for Automatic In-line Quantitative Myocardial Perfusion Mapping

Quantitative perfusion (QPerf) is an emerging technique to directly measure blood supply to the myocardial mussel. It is demonstrating potential to be a valid clinical tool for ischemia and many non-ischemic cardiac disease. 

This repo contains the cardiac flow mapping tool we developed, in supplement to our paper submitted to [Magnetic Resonance in Medicine](https://onlinelibrary.wiley.com/journal/15222594), titled "Automatic In-line Quantitative Myocardial Perfusion Mapping: processing algorithm and implementation". This paper is currently under review.

The Blood-tissue-exchange model (BTEX [1](by Prof. James Bassingthwaighte, https://www.physiome.org/)) is used in this study for quantitative perfusion. For fully disclosure, there are many different models proposed in past two decades [2](https://www.ncbi.nlm.nih.gov/pubmed/22173205). 

BTEX model (20 version) solves two partial differential equations for myocardial blood flow (MBF, ml/min/g), permeability-surface-area product (PS, ml/min/g), blood volume (V_b, ml/g) and interstitial volume (V_isf, ml/g).

![BTEX QPerf](./images/BTEX_pixel_size_mapping.JPG "Pixel-wise BTEX flow mapping for perfusion")

To perform perfusion fitting and estimate 
