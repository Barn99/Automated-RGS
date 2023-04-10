# Automated-RGS
Automated-RGS is a machine learning approach for the Rat Grimace Scale (RGS). Currently machine learning approaches exist for the capturing of rodent facial images and the grading for the Mouse Grimace Scale (MGS). However, depsite their widespread useage in pain research, no current systems have been developed to fully automate the RGS grading process. This program aims to solve this gap in research by fine-tuning Ultralytic's YOLOv5 for facial action unit detection and huggingface's Vision Transformer (ViT) for action unit pain grading.

<details open>
<summary>Install</summary>
Clone repo and install requirements.txt
  
```
 git clone https://github.com/Barn99/Automated-RGS
 cd Automated-RGS
 pip install -r requirements.txt
```

The file size for the transformer weights are too large and need to be download seperately ([ViT Weights](https://drive.google.com/drive/folders/1Cl_5GyouX7sDLv1NUKuq_YxrrQRQMYKn?usp=sharing/).
 
Vision Transformer Weights:
https://drive.google.com/drive/folders/1Cl_5GyouX7sDLv1NUKuq_YxrrQRQMYKn?usp=sharing
