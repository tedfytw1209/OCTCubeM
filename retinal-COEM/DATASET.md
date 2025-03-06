
## General
We provide description of our inhouse dataset structure to help to run retinal-COEP. To run this task, you may need to have a paired datset.

## AI-READI Dataset

### Downloading the AI-READI Retinal OCT Dataset

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

We aim to provide the evaluation code for v2.0.0 dataset and the related results measurement in the future. We also plan to release a 3-modalities model trained on AI-READi (OCT, FAF, IR) triplets for better understanding of our Tri-COEP design for OCTCube-EF, stay tuned! 🚀


## Inhouse Dataset

We provide the format of our in-house dataset structure to facilitate the organization of the pretraining dataset. You can find the dummy file examples under `OCTCubeM/assets/Oph_cls_task/`.
Below is an example directory structure for a **Heidelberg Spectralis 61-frame volume** stored in PNG format:


```
Ophthal/patient_id/macOCT/session_id/oct_000.png
Ophthal/patient_id/macOCT/session_id/oct_001.png
Ophthal/patient_id/macOCT/session_id/oct_002.png
...
Ophthal/patient_id/macOCT/session_id/oct_060.png
Ophthal/patient_id/macOCT/session_id/ir.png
Ophthal/patient_id/macOCT/session_id/oct.json (optional, storing scan position and pixel spacing, etc)
Ophthal/patient_id/macOCT/session_id/ir.json (optional, storing pixel spacing, etc)
```

For an example of metadata, please refer to `OCTCubeM/assets/Oph_cls_task/`. By specifying the correct prefix, you should be able to load the dummy metadata.

### Loading the Dataset

- The in-house dataset can be loaded using the `PatientDataset3D_inhouse()` class in `PatientDataset_inhouse.py`.
- This class provides flexible arguments for parsing **patient ID, session ID, laterality**, and other useful metadata.

### Iteration Modes

There are **two default iteration modes**:

1. **Patient Mode**
   - Recommended when a patient has only one instance in the dataset.

2. **Visit Mode** *(default for large-scale pretraining datasets)*
   - Recommended when a patient might have data from **both eyes** or **multiple sessions**.

### Metadata Handling

The `PatientDataset2DCenter_inhouse()` class provides multiple metadata processing functions.
We categorize metadata into three essential types for a **successful dataset instance**:

1. **Disease Information**
2. **Patient Information**
3. **OCT Data Instance Information**

To organize your metadata, follow the format below:
```
split_path/
----train_pat_list.txt
--------patient_id_0
--------patient_id_1
...
----val_pat_list.txt
--------patient_id_n
--------patient_id_n+1
...
----test_pat_list.txt
--------patient_id_N
--------patient_id_N+1
...
```

#### Example: `multilabel_cls_dict.json`
```json
{
  "disease_list": {
    "None": 0,
    "DME": 1,
    "AMD": 2,
    "POG": 3,
    "PM": 4,
    "ODR": 5,
    "VD": 6,
    "CRO": 7,
    "RN": 8
  },
  "patient_dict": {
    "patient_id_0": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    ...
  }
}
```


#### Example: `patient_dict_w_metadata_first_visit_from_ir.pkl`
```json
{'instance_id_0': {'ptid': 'patient_id_0', 'study': 'OCT', 'series': 'Volume IR', 'age': 73.8042, 'laterality': 'L', 'instance': 'instance_id_0', 'type': 'OCT', 'imshape': [61, 496, 512]},
...
}
```

### Metadata Recommendations

We highly recommend following the structured formats outlined above when generating metadata for your dataset.
Properly formatted metadata improves dataset usability and ensures smooth dataset loading.

By structuring disease information, patient metadata, and dataset splits correctly, you can:

- Facilitate efficient data retrieval and processing.
- Ensure compatibility with dataset loading scripts.
- Maintain consistency across different datasets and experiments.

Adhering to these metadata guidelines will help streamline dataset integration and improve overall research workflow. 🚀


## Building your own 3-modality dataset
We provide a bunch of useful utils to help you build your own 3-modality dataset. Please check `dataset_management.py` and `multimodal_dataset.py` in `OCTCubeM/retinal-COEM/src/training/` for more details.

Specifically, `dataset_management.py` provides the `oph_dataset()` classes for you to load your own index file and multiple types of fundus imaging format, including: `.mhd`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.dcm`.

`multimodal_dataset.py` provides the `OphthalDataset()` class for you to load flexible types of your 3-modalities data. `OCTFAFIRClsDataset()` is also provided for you to load your 3-modality data with classification labels, which can be used for fine-tuning the pre-trainid multi-modal foundation model.