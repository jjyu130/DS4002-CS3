# Weather Event CNNs Case Study


## 1. Software and Platform
- **Platform Used:** Mac was used to write and run scripts. The scripts can be run across platforms.
- **Software Used**: Python 3
 
- **Add-on Packages:**  
  - `pathlib` – filesystem paths and directory handling
  - `shutil` – moving, copying, and deleting files during preprocessing
  - `random` – random sampling for dataset splitting
  - `numpy` – numerical operations
  - `PIL` – image loading, conversion, resizing, preprocessing  
  - `tensorflow` – end-to-end deep learning framework
    - `keras` → (`layers`, `models`, `optimizers`, `callbacks`)
      - `applications` → (`InceptionV3`, `preprocess_input`)
  - `torch` - tensor computation and auto-grad framework
  - `torchvision` - datasets, transforms, pretrained vision models
  - `tqdm` - quick, customizable progress bar utility
  - `scikit-learn` - classical machine learning algorithms library
  - `seaborn` - machine learning models and metrics
  - `matplotlib` - plotting and visualization
    

## 2. Documentation Map
The hierarchy of folders and files contained in this project are as follows:

```text
DS4002-Project3
├── DATA
│   ├── dataset_split/               # Final 80/10/10 train–val–test split for modeling
│   │   ├── train/
│   │   │   ├── cellconvection/
│   │   │   ├── duststorm/
│   │   │   ├── hurricane/
│   │   │   ├── rollconvection/
│   │   │   └── wildfires/
│   │   ├── val/
│   │   │   ├── cellconvection/
│   │   │   ├── duststorm/
│   │   │   ├── hurricane/
│   │   │   ├── rollconvection/
│   │   │   └── wildfires/
│   │   └── test/
│   │       ├── cellconvection/
│   │       ├── duststorm/
│   │       ├── hurricane/
│   │       ├── rollconvection/
│   │       └── wildfires/
│   ├── cleaned_data/                # Standardized RGB images after preprocessing
│   │   ├── cellconvection/
│   │   ├── duststorm/
│   │   ├── hurricane/
│   │   ├── rollconvection/
│   │   └── wildfires/
│   ├── raw_weather_images/          # Original satellite imagery from Harvard Dataverse
│   │   ├── cellconvection/
│   │   ├── duststorm/
│   │   ├── hurricane/
│   │   ├── rollconvection/
│   │   └── wildfires/
│   └── README.md
├── SCRIPTS
│   ├── 01_preprocess_and_split.py 
│   ├── 02_InceptionV3.py 
│   ├── 03_InceptionV3_confusion_matrix.py 
│   └── 04_resnet-50.py
├── SUPPLEMENTAL_RESOURCES
│   ├── What is Transfer Learning.pdf
│   ├── Is Climate Change Increasing Disaster Risk.pdf
├── Hook_Document.pdf
├── WeatherEventCNN-Rubric.pdf
├── LICENSE.md
└── README.md
```

## 3. Reproducing Our Results
  1. **Set up Python and install required add-on packages**
     - Clone this repository: https://github.com/jjyu130/DS4002-Project3/
     - Ensure you have Python 3 installed on your system.
     - See section 1 for packages needed.
  2. **(Recommended) Use Git LFS.**
     - You can directly download the dataset, but it is recommended to use Git LFS to load the large data files directly from GitHub.
     - To properly load the image dataset (once the repository is cloned): 
       1. Install Git LFS on your local system at https://git-lfs.com/
       2. Initialize Git LFS by running “git lfs install”
       3. Go to your local Git repository’s root
       4. Run “git lfs pull”
  3. **(Optional) Prepare the dataset**
     - Run `01_preprocess_and_split.py` from the `SCRIPTS` folder, which creates
          `cleaned_data` by processing `raw_weather_images` and splits into `dataset_split`
          in the `DATA` folder.
     - Otherwise, proceed with preprocessed data, which is already provided.
4. **Run model training script**
     - Navigate to the `SCRIPTS` folder.
     - Scripts 2-4 build, train, and evaluate 2 CNN models on the prepared `DATA/dataset_split/` directory.
       Models and evaluation metrics are saved to `OUTPUT` folder.
5. **(Recommended) Use UVA Rivanna.**
     - The two models take a long time to train on a personal local machine. Utilizing the UVA HPC Rivanna Systems is highly recommended to reduce training and testing time from hours to within ten minutes.
     - To properly upload the repository onto Rivanna:
       1. Login to UVA Rivanna through OnDemand
       2. Go to your Scratch/YOUR_COMPUTING_ID folder and click on "Open in Terminal"
       3. Clone the repository onto this folder.
       4. Choose the dataset you want to work with ("raw_weather_images","cleaned_data")
       5. The images in the folder are pointers files, not real images. Replace the images in the given dataset with the real images by deleting image files and uploading the real images. 
          - This is recommended to be done folder by folder ("cleaned_data/wildfires") as OnDemand can't upload high volumes of data in a single try.
       6. Create an interactive session in JupyterLab with GPU-MIG for high performance.
       7. Proceed to run the scripts through a JupyterLab notebook. 


 

     
