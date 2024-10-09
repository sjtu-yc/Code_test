# Code of IDFree-PER
**Project Structure**
- classification/ *(Code on classification dataset)*
  - data/ *(Dataset and pre-process scripts)*
  - model/
  - visualization/ *(Visulazation of results)*
  - central_train/ *(Centralized training without personalization)*
  - device_train/ *(Training on device)*
    - nodes/ *(clients)*
    - device_train_prefix.py *(Training embeddings on devices)*
    - ondevice_baseline.py *(Ondevice training baseline of our paper)*
  - per_train/ *(Centralized personalized-training)*
- generation/ *(Code on generation dataset)*
  - ...

**Running Steps**

1. Pre-process dataset: run the script in *data/* folder
2. Centralized training with no personalized information: *central_train/train_cloud.py*
3. Training embeddings on devices based on the centralized model: *device_train/device_train_prefix.py*
4. Centralized personalized-training based on centralized model and user embeddings: *per_train/train_cloud.py*