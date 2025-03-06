## OCTCube - Public Datasets Benchmark
We proivde an detailed instruction for obtaining the public datasets used in OCTCube.

### Duke14
| Dataset | Download Link |
| ------------- | ------------------ |
| Duke14 | [Download page](https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm) |

After downloading, please unzip the dataset and place it in the `OCTCubeM/assets/ext_oph_datasets/DUKE_14_Srin/` directory. Then run the scripts in `OCTCubeM/assets/ext_oph_datasets/DUKE_14_Srin/extract_duke14_data.ipynb` to extract the data.

### OIMHS
| Dataset | Download Link |
| ------------- | ------------------ |
| OIMHS | [Download page](https://springernature.figshare.com/collections/OIMHS_An_Optical_Coherence_Tomography_Image_Dataset_Based_on_Macular_Hole_Manual_Segmentation/6662624/10) |

After downloading, please unzip the dataset and place it in the `OCTCubeM/assets/ext_oph_datasets/OIMHS_dataset/` directory.

First run `get_split_and_index_csv.ipynb` and then run the scripts in `OCTCubeM/assets/ext_oph_datasets/OIMHS/extract_oimhs_data.ipynb` to extract the data.

### UMN
| Dataset | Download Link |
| ------------- | ------------------ |
| UMN | [Download page](https://people.ece.umn.edu/users/parhi/.DATA/) |

After downloading, please unzip the dataset and place it in the `OCTCubeM/assets/ext_oph_datasets/UMN/` directory. Then run the scripts in `OCTCubeM/assets/ext_oph_datasets/UMN/extract_UMN.ipynb` to extract the data.

### HCMS
| Dataset | Download Link | Utils |
| ------------- | ------------------ | ------------------ |
| HCMS | [Download page](https://iacl.ece.jhu.edu/index.php?title=Resources) | [Preprocessing code (MATLAB required)](https://github.com/heyufan1995/oct_preprocess/tree/master)

First download the dataset and get the preprocessing code. Place the data at `OCTCubeM/assets/ext_oph_datasets/HCMS`, Replace the prefix in `Scripts/hc/filename.txt` with your file paths (should be absolute path). Then run the scripts in `Scripts/generate_hc_train.m/` to extract the data, the resulting images will be stored under `image/` directory. Place it at `OCTCubeM/assets/ext_oph_datasets/HCMS/`, and then run `OCTCubeM/assets/ext_oph_datasets/HCMS/process_hcms_data.ipynb` to extract the data.


### SLIViT
| Repo | Model Link | Echo Dataset |
| ------------- | ------------------ | ------------------ |
| [SLIViT](https://github.com/cozygene/SLIViT) | [Download model](https://drive.google.com/open?id=1f8P3g8ofBTWMFiuNS8vc01s98HyS7oRT) | [Download Echo dataset](https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a) |

1. At `OCTCubeM/assets/SLIViT/`, run the following to get the oct pre-trained weights:
```bash
pip install gdown
gdown --folder https://drive.google.com/open?id=1f8P3g8ofBTWMFiuNS8vc01s98HyS7oRT
```

2. To try OCTCube on Echo dataset, after downloading it, run `convert_avi_to_tiff.py` and `get_echonet_csv.py` to convert the data to the format we used in our paper. Then, replace the `OCTCubeM/OCTCube/assets/us3d_meta/echonet.csv` with the new one.

3. There is no extra preparation to try OCTCube on NoduleCT3D dataset, directly run the script will stor the dataset under `OCTCubeM/assets/medmnist_data/nodulemnist3d_64.npz`.


### GLAUCOMA
| Dataset | Download Link |
| ------------- | ------------------ |
| GLAUCOMA | [Download page](https://zenodo.org/records/1481223) |

After downloading, please unzip the dataset and place it in the `OCTCubeM/assets/ext_oph_datasets/GLAUCOMA/` directory. Then run the scripts in `OCTCubeM/assets/ext_oph_datasets/process_glaucoma.sh` to extract the data.

### AI-READI
| Dataset | Download Link |
| ------------- | ------------------ |
| AI-READI | [Download page](https://aireadi.org/) |

#### Step 1: Visit the Dataset Documentation
Go to the dataset documentation page:
🔗 [AI-READI Retinal OCT Dataset](https://docs.aireadi.org/docs/2/dataset/retinal-oct/)

#### Step 2: Review Dataset Details and access the Dataset on FAIRhub

The dataset is hosted on **FAIRhub**. Visit:  🔗 [FAIRhub Dataset Page](https://fairhub.io/datasets/2) and apply to get access.

**NOTE:** We used the v1.0.0 version to conduct the experiment in our OCTCubeM paper. If you find the v1.0.0 dataset is no longer accessible, we provide the patient id list used for reproducing our experiment: `OCTCubeM/assets/aireadi_v1_patient.txt`

#### Step 3: Download the Dataset and Set Up the Directory Structure
In OCTCubeM paper, AI-READI dataset is set to be used for evaluation, if you want to test our model on AI-READI dataset, please follow the directory structure below:

In `OCTCubeM/retinal-COEM/src/scripts/retclip_eval/reticlip_eval_aireadi_example.sh`, modifies the `data_path` with your AI-READI dataset path.

If you have downloaded the AI-READI v2.0.0 dataset, the retrieval results will be tested on the v2.0.0 dataset, which might result in slightly different results, but the difference should be minor.


### In-house Data preparation (See Oph_cls_task/)

Under this folder, we provide the example of all the necessary meta data with the exact same strcuture we used for our UW-Oph data experiments with dummy data. You can use this as a template to prepare your own data. Check `OCTCubeM/OCTCube/util/PatientDataset_inhouse.py` for more details.