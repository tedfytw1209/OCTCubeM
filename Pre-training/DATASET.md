# Dataset Preparation


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

For an example of metadata, please refer to `OCTCubeM/assets/Oph_cls_task/`. By specifying the correct prefix, you should be able to load the dummy metadata.

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

Adhering to these metadata guidelines will help streamline dataset integration and improve overall research workflow.  🚀


---


## Kermany Dataset

In our prepared code, we also incorporated a 2D OCT frame pre-training pipeline that can split the in-house 3D OCT volumes into 2D with the option to also include external datasets. Here, we use Kermany dataset as an example.

The Kermany dataset can be downloaded from [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3).

### Usage Instructions

1. **Setting Up the Data Path**  
   After downloading the dataset, update the `kermany_data_dir` variable with your local path to the dataset. You can find the default value in `main_pretrain_oph_joint_2d512_flash_attn.py` for reference.

2. **Loading the Dataset**  
   The Kermany dataset (or any other 2D dataset you wish to use as a replacement) can be loaded using the `PatientDatasetCenter2D_inhouse_pretrain()` class, which is implemented in `PatientDataset_pretrain.py`.

3. **Training Limitations**  
   Currently, standalone training using only the Kermany dataset (i.e., 2D training) is not supported. You will need an additional in-house dataset to successfully run the training script.
