import cv2
import typing
import shutil
import os
import pandas as pd
from wordSegmentation import wordSegmentation, prepareImg
import numpy as np
from modules.modelconfigs import BaseModelConfigs
from modules.inferenceModel import OnnxInferenceModel
from modules.text_utils import ctc_decoder, get_cer, get_wer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def get_paths_and_texts(partition_split_file):
    paths_and_texts = []
    
    with open (partition_split_file) as f:
            partition_folder = f.readlines()
    partition_folder = [x.strip() for x in partition_folder]
    
    with open ('Datasets/IAM_Words/words.txt') as f:
        for line in f:
            if not line or line.startswith('#'): 
                continue
            line_split = line.strip().split(' ')
            assert len(line_split) >= 9
            status = line_split[1]
            if status == 'err': 
                continue

            file_name_split = line_split[0].split('-')
            label_dir = file_name_split[0]
            sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
            fn = '{}.png'.format(line_split[0])
            img_path = os.path.join('Datasets/IAM_Words/words', label_dir, sub_label_dir, fn)
            gt_text = ' '.join(line_split[8:])
            if len(gt_text)>16:
                continue

            if sub_label_dir in partition_folder:
                paths_and_texts.append([img_path, gt_text])
    return paths_and_texts

if __name__ == "__main__":
    configs = BaseModelConfigs.load("Models/word_handwriting_recognition/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    test_data = get_paths_and_texts('Models/word_handwriting_recognition/test.uttlist')
    accum_cer = []
    accum_wer = []
    for image_path, label in test_data:
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
        accum_cer.append(cer)
        accum_wer.append(wer)
        image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"Average CER: {np.average(accum_cer)}")
    print(f"Average WER: {np.average(accum_wer)}")

    
    # import random
    
    # df = pd.read_csv("Models/word_handwriting_recognition/line_val.csv").values.tolist()
    # random.shuffle(df)
    # image_paths = []
    # for file in glob.glob(os.path.join('Images','*')):
    #     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    #         image_paths.append(file)
    
    # # Test line recognition using Scale space algothrm to segmentation        
    # for line in df:
    #     img = prepareImg(cv2.imread(line[0]), 80)
    #     img_cp = img.copy()
    #     res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
    #     if not os.path.exists('tmp'):
    #         os.mkdir('tmp')
    #     for (j, w) in enumerate(res):
    #         (wordBox, wordImg) = w
    #         (x, y, w, h) = wordBox
    #         cv2.imwrite('tmp/%d.png'%j, wordImg)
    #         cv2.rectangle(img_cp,(x,y),(x+w,y+h),(0,255,0),1) # draw bounding box in summary image
        
    #     cv2.imshow("Image", img_cp)
    #     imgFiles = os.listdir('tmp')
    #     len_imgFile = len(imgFiles)
    #     pred_line = []
    #     for i in range(len_imgFile):
    #         image = cv2.imread('tmp/' + str(i) + '.png')
    #         prediction_text = model.predict(image)
    #         print(prediction_text)
    #         pred_line.append(prediction_text)
    #     print('[Prediction]: '+ str(' '.join(pred_line)))
    #     print('[Label]: '+ str(line[1]))
    #     print('[CER]: ' + str(get_cer(str(' '.join(pred_line)), str(line[1]))))
    #     print('[WER]: ' + str(get_wer(str(' '.join(pred_line)), str(line[1]))))

    #     shutil.rmtree('tmp')
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        
    
        
