# Automated-RGS
Automated-RGS is a machine learning approach for the Rat Grimace Scale (RGS). Currently machine learning approaches exist for the capturing of rodent facial images and the grading for the Mouse Grimace Scale (MGS). However, depsite their widespread useage in pain research, no current systems have been developed to fully automate the RGS grading process. This program aims to solve this gap in research by fine-tuning Ultralytic's YOLOv5 for facial action unit detection and huggingface's Vision Transformer (ViT) for action unit pain grading. This model use 3/4 existing RGS action units and does not consider whisker changes. Currently the frontal-facing images are usually developed using Rodent Face Finder®, however, a new program using the YOLOv5 object detection is being developed to upgrade the speed and accuracy of the existing program.

<details open>
<summary>Install</summary>
Clone repo and install requirements.txt. This program has only been tested using python 3.10.
  
```
 git clone https://github.com/Barn99/Automated-RGS
 cd Automated-RGS
 pip install -r requirements.txt
```

The file size for the transformer weights are too large and need to be download seperately ([ViT Weights](https://drive.google.com/drive/folders/1Cl_5GyouX7sDLv1NUKuq_YxrrQRQMYKn?usp=sharing/)). 
 Download the weights and place them in the weights folder (Automated-RGS/Weights).
 
</details>

<details>
<summary>Useage</summary>

```
python Automated_RGS.py -i [File Directory with Images] -o [Output Filename].xlsx 
```
Optional useage: -c (confidence): Change the confidence threshold for YOLOv5 detection model (default:0.25)
An excel file with the image name, RGS score, orbital tightening score, ear changes score, and nose flattening score will appear in the output file.

</details>

## References
If you use this corpus, please cite the following paper:

Paper currently being written. Will be updated once published.
