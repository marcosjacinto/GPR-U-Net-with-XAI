# Karstified Zone Interpretation Using Deep Learning Algorithms: Convolutional Neural Networks Applications and Model Interpretability with Explainable AI


## Abstract

The Ground Penetration Radar (GPR) can be used to assist in mapping karstified zones in analogues for the characterization and understanding of carbonate reservoirs. With the aid of GPR it is possible to understand the behavior of karstification processes in carbonates, and thus expand the knowledge to the reservoir level. In this context, this study seeks to develop Deep Learning models based on Convolutional Neural Networks, using the U-Net architecture, capable of assisting in the mapping of karstified zones imaged through GPR surveys. Moreover, Explainable Artificial Intelligence (XAI) techniques using SHapley Additive exPlanation (SHAP) values are applied to promove interpretability and explainability of the generated models. These techniques were employed in order to assess the rules found by the models, modeling quality and the presence of biases in the model. Moreover, distinct settings with regard to background SHAP values were tested and compared to assess how they influence model explainability. Through the SHAP values, it was possible to notice that the Energy attribute was the feature that provided more information in the modeling, and consequently, provided a greater weight in the model rules, while the other features presented a less relevant contribution. Furthermore, the type of sampling used to define reference values for the SHAP values resulted in different interpretations for the contributions of the features. Finally, it was possible to generate a model capable of aiding in mapping karstified, as well as using an extremely important technique to promote the understanding of complex models and to allow greater cooperation between experts in the geosciences and results generated through Deep Learning techniques.

## About the model

We use tensoflow in order to create the models whose script can be found in model/train.py. The model receives as input a tensor with a dimension of 16x16xNc. Where Nc is the number of channels which in the case of the complete model is 6: GPR Section, Similarity, Energy, Instantaneous Phase, Instantaneous Frequency and Hilbert Trace/Similarity. The output is a binary 16x16 image, which represents an interpreted section of karstified zones.

## About Shap values

**SHAP (SHapley Additive exPlanations)** is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

## Folder structure

- data:
    - original: contains the attributes in .dat format, the gpr sections in .SGY and the ground truth interpretations as an image ;
    - processed: contains the numpy arrays created from the processing of the original data, as well as a .pkl object that contains the Yeo-Johnson power transformer;
- model: 
    - figures_history: figures that show the training history, metrics and loss function;
    - models: contains the models created and presented in the paper as .h5 files;
- notebooks: jupyter notebooks which exemplifies some of the steps taken during this research, such as using SHAP values;
- xai:
    - figures: contains the explanation created using SHAP as figures, some of those are present in the paper;
    - results: overall results and parts of gpr sections used in this paper to create explanations using SHAP;

## Data availability
We use the [DVC](http://dvc.org/) package to control the versioning of the data and to keep references to the data used on GitHub, since we can't make it publicly available due to data confidentiality.