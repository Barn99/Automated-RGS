import os
import sys
import torch
import logging
import argparse
import pandas as pd
from PIL import Image
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from RGS_functions import load_models, Average

#Check available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device.type}")

#Create argument parser
parser = argparse.ArgumentParser(description='Process input and output file paths.')
parser.add_argument('-i', '--input', type=str, help='Input file path', required=True)
parser.add_argument('-o', '--output', type=str, help='Output file name (include filename.xlsx)', required=True)
parser.add_argument('-c;', '--confidence', type=float, help='YOLO model confidence threshold', default=0.25)
args = parser.parse_args()

###Load the Models###
print('Loading Models...')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

confidence = args.confidence

obj_model, feature_extractor, eye_model, ear_model, nose_model = load_models(confidence, device)



###File Path###
file_path = args.input

###Preprocess Data###
print('Preprocessing Images...')
image_data = []
for image in os.listdir(file_path):
    f = os.path.join(file_path, image)
    if (image.endswith(".png")) or (image.endswith(".jpg")) or (image.endswith(".PNG")) or (image.endswith(".JPG")):
        pil_im = Image.open(f)
        pil_im = pil_im.convert('RGB')
        width, height = pil_im.size
        image_data.append([image, pil_im, width, height])


###Analyze Images###
print('Analyzing the Images...')

RGS = []

#Run through each image in the folder
for file, image, width, height in image_data:
    print(file)
    
    #run image in YOLO model
    yolo_out = obj_model(image)
    features = yolo_out.xyxyn[0].numpy()
    
    eye = []
    ear = []
    nose = []
    
    for detect in features:
        bbox = []
        #seperate into class and bbox cords
        bbox = detect[0:4]
        conf = detect[-2]
        class_id = detect[-1]
        
        #resize boxes to original image size
        bbox[0] = bbox[0]*width
        bbox[1] = bbox[1]*height
        bbox[2] = bbox[2]*width
        bbox[3] = bbox[3]*height
        
        #sort images for correct ViT model
        if class_id == 0:
            crop_img = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            ear.append((conf, crop_img))
            
        elif class_id == 1:
            crop_img = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            eye.append((conf, crop_img))
            
        elif class_id == 2:
            crop_img = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            nose.append((conf, crop_img))
            
    #only take max conf 2 ears and eyes, and 1 nose for each image
    ear.sort(reverse=True, key=lambda x: x[0])
    eye.sort(reverse=True, key=lambda x: x[0])
    nose.sort(reverse=True, key=lambda x: x[0])
    ear = [img for conf, img in ear[:2]]
    eye = [img for conf, img in eye[:2]]
    nose = [img for conf, img in nose[:1]]

    rgs_eye = []
    rgs_ear = []
    rgs_nose = []
    
    #run through ViT model
    for feature in eye:
        inputs = feature_extractor(feature, return_tensors='pt')
        
        with torch.no_grad():
            logits = eye_model(**inputs).logits
            pred = logits.argmax(-1).item()

            rgs_eye.append(pred)
            
    for feature in ear:
       
        inputs = feature_extractor(feature, return_tensors='pt')
        with torch.no_grad():
            logits = ear_model(**inputs).logits
            pred = logits.argmax(-1).item()

            rgs_ear.append(pred)

    for feature in nose:
        
        inputs = feature_extractor(feature, return_tensors='pt')
        with torch.no_grad():
            logits = nose_model(**inputs).logits
            pred = logits.argmax(-1).item()
            rgs_nose.append(pred)
            
    #score RGS
    RGS_score = (Average(rgs_eye) + Average(rgs_ear) + Average(rgs_nose))/3
        
    
    RGS.append([file, RGS_score, Average(rgs_eye), Average(rgs_ear), Average(rgs_nose)])

###Save Results###
    
#DataFrame from the list
df = pd.DataFrame(data=[row for row in RGS], 
                  columns=['Filename', 'RGS', 'Orbital Tightening', 'Ear Changes', 'Nose Flattening'])

# Save the DataFrame to an Excel file
output_path = os.path.join('output', args.output)
df.to_excel(output_path, index=False)

print('Finished!')


