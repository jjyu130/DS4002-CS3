# Investigating Reproducibility of Weather Event Image Classifier


## 1. Software and Platform
- **Platform Used:** Mac was used to write and run scripts. The scripts can be run across platforms.
- **Software Used**: Python 3
 
- **Add-on Packages:**  
  - `pandas` - data loading and manipulation
  - `numpy` - numerical operations
  - `matplotlib` – plotting and visualization

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
├── OUTPUT
│   ├── RGB_by_class.png   
│   ├── contrast_by_class.png
│   ├── evaluation_inceptionv3_best.png
│   ├── inceptionv3_best.keras
│   └── inceptionv3_final.png  
├── SCRIPTS
│   ├── 01_preprocess_and_split.py 
│   └── 02_InceptionV3.py 
├── LICENSE.md
└── README.md
```

## 3. Reproducing Our Results
  1. **Set up Python and install required add-on packages**
     - Clone this repository: https://github.com/jjyu130/DS4002-Project3/
     - Ensure you have Python 3 installed on your system.
     - See section 1 for packages needed.
     
 

     
