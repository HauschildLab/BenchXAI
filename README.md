<div align="center">
  <img width="550" src=images/BenchXAI_Logo.png>
<h3><b>A large BENCHmarking study of XAI methods</b></h3>
With PyTorch
</div>

## Overview
A large evaluation study of XAI methods on biomedical data is needed since biomedical data is divers and an XAI method 
effective for one datatype may not be effective for others. That is why we implemented our own XAI evaluation framework, 
called BenchXAI. We use it in our Paper () to evaluate the performance of 15 different XAI methods on three different tasks, 
namely medical data, medical image and signal data, and biomolecular data. We present a guideline for researchers which will 
help in the selection of an appropriate XAI method for their specific tasks. This will eliminate unnecessary testing of different 
XAI methods and thus speed up the process of opening the 'black-box' in the future. 

The BenchXAI pacakge is mainly build using [PyTorch](https://pytorch.org/), 
[Schikit-learn](https://scikit-learn.org/stable/) and [Captum](https://captum.ai/).

#### Installation
The package can simply be installed using pip with the source files in the [dist folder](/dist/). 
````commandline
pip install dist\benchxai-0.0.1.tar.gz
````
or 
````commandline
pip install dist\benchxai-0.0.1-py3-none-any.whl
````
Or it can be build from the source code using 
````commandline
python -m build
````

#### How does the benchmark work
A simplified workflow of the benchmark can be seen in the following workflow:
In the first step, datasets, machine learning models (ML), deep learning models (DL), and XAI methods are loaded. 
In the second step our BenchXAI framework is used to split some samples of each class from the dataset for testing 
and then perform a Monte Carlo Cross Validation (MCCV) followed by the application of 15 different XAI methods on the 
test data using the trained DL models. In step 3 of our BenchXAI framework all results of step 2 are saved. 
The fourth and last step is the evaluation of all results.

<div align="center">
  <img width="850" src=images/BenchXAI_Workflow.png>
</div>

Example code for running the benchmark can be found in the [examples folder](/examples/).
