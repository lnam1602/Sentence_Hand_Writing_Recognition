import cv2
import glob
import os
import typing
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from modules.modelconfigs import BaseModelConfigs
from modules.inferenceModel import OnnxInferenceModel
from modules.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer
import matplotlib.pyplot as plt

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

if __name__ == "__main__":
    configs = BaseModelConfigs.load("Models/sentence_handwriting_recognition/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/sentence_handwriting_recognition/val.csv").values.tolist()
    random.shuffle(df)

    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image: ", image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)
        print(f"CER: {cer}; WER: {wer}")

        accum_cer.append(cer)
        accum_wer.append(wer)

        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        cv2.imshow(prediction_text, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")

    
    

    
    
            
    
    