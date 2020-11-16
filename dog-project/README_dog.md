[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/Sample_App_Screen_Shot.jpg "Loading App"
[image5]: ./images/Sample_App_Screen_Shot_Image_Selected.jpg "Image Selection"
[image6]: ./images/Sample_App_Screen_Shot_Output.jpg "App Output"

## Content:

- [Overview](#overview)
- [Instructions](#instructions)
- [Instructions  -  to run Jupyter Notebook](#inst_ipnoteb)
- [Instructions -  to run web app](#webapp)
  - [Optional Steps](#opt)
- [Project Jupyter Notebook](#open_nb)
  - [ETL Pipeline](#etl_)
  - [ML Pipeline](#ml_)
  - [Flask App](#flask)
- [Results](#results)
- [Project File Structure](#files)
- [Requirements](#req)
- [Mentions](#mentions)
- [References](#ref)


<a id='overview'></a>

## Dog Breed Classifier  -  Project Overview

This project provides the user with the ability to predict the breedof a dog from an image selected from their computer.  It consists of 2 part:
1. The derivation of a convolutional neural network to predict the dog breed.  tis portion is using jupyter notebook.
2. A simple web application that allows the user to select an image which is the previewed on the page and upon submitting the image the dog breed will be predicted and displayed on the page. 

The algorithm built will, given an image of a dog, provide an estimate of its breed.  If supplied an image of a human, the code will identify the resembling dog breed. Lastly, If provided an image that it can not classify as dog or human it will provide an output indicating it can not provide an estimate for something not from this world.
 
<p align="center">
<img class="center-text" src="./images/Sample_App_Screen_Shot.jpg" alt="drawing" width="50%"/>
<img class="marginauto" src="./images/Sample_App_Screen_Shot_Image_Selected.jpg" alt="drawing" width="50%"/>
<img class="marginauto" src="./images/Sample_App_Screen_Shot_Output.jpg" alt="drawing" width="50%"/>
</p>

<a id='instructions'></a>

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
# Create a directory on yur computer, so Git doesn't get messy, and go to it
mkdir my-dir 
cd my-dir

# Start a Git repository
git init

# Track repository, do not enter subdirectory
git remote add -f origin https://github.com/43piRcubed/portfolio

# Enable the tree check feature
git config core.sparseCheckout true

# Create a file in the path: .git/info/sparse-checkout
# That is inside the hidden .git directory that was created
# by running the command: git init
# And inside it enter the name of the sub directory you only want to clone
echo 'files' >> .git/info/sparse-checkout

## Download with pull, not clone
git pull origin master
```

<a id='inst_ipnoteb'></a>

### Instructions  -  to run Jupyter Notebook

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The direcotry already exists but is empty 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. The directory already exists but is empty

4. Donwload Bottleneck Features and place them into the repo, at location `path/to/dog-project/bottleneck_features`.  The directory already exists but is empty.
	1.	[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.
	2.	[Resnet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) for the dog dataset.
	3.	[Xception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) for the dog dataset.

5. Optional Steps 

<a id='opt'></a>

	1. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__:

    - follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  
    - If you are using an EC2 GPU instance, you can skip this step.

	2. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```
	
	3. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.6
	source activate dog-project
	pip install -r requirements/requirements.txt
	```  
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.6
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
	4. (Optional) **If you are using AWS**, install Tensorflow.

 	```
	sudo python3 -m pip install -r requirements/requirements-gpu.txt
	```
	
	5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.

	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```

	6. (Optional) **If you are running the project on your local machine (and not using AWS)**:

    - create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
	```
	python -m ipykernel install --user --name dog-project --display-name "dog-project"
	```

<a id='open_nb'></a>

6. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

  - (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

## Project Submission

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:
- The `dog_app.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.
- Any additional images used for the project that were not supplied to you for the project. __Please do not include the project data sets in the `dogImages/` or `lfw/` folders.  Likewise, please do not include the `bottleneck_features/` folder.__

Alternatively, your submission could consist of the GitHub link to your repository.