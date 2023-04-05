import os
import logging
import torch
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification

def load_models(confidence, device):
    
    #YOLO Model
    obj_model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path = "Weights/YOLO_weights.pt");

    #Confidence threshold
    obj_model.conf = confidence;

    #Load ViT Models
    feature_extractor = ViTFeatureExtractor.from_pretrained(
                'google/vit-base-patch16-224-in21k');

    eye_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',num_labels = 3);

    ear_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',num_labels = 3);

    nose_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',num_labels = 3);

    eye_model.load_state_dict(torch.load("Weights/eye_weights.bin", map_location=device))
    eye_model.eval()


    ear_model.load_state_dict(torch.load("Weights/ear_weights.bin", map_location=device))
    ear_model.eval()


    nose_model.load_state_dict(torch.load("Weights/nose_weights.bin", map_location=device))
    nose_model.eval()

    return obj_model, feature_extractor, eye_model, ear_model, nose_model


def Average(lst):
    if len(lst) == 0:
        return float('NaN')
    return sum(lst)/len(lst)
