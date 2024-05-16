import cv2
import os
import typing
import numpy as np
from modules.modelconfigs import BaseModelConfigs
from modules.inferenceModel import OnnxInferenceModel
from modules.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer
import re
import shutil

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
    
    image_path = 'image.png'
    output_dir = 'tmp'
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cp = gray.copy()

    # Áp dụng ngưỡng nhị phân để tạo ảnh đen trắng
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Sử dụng phép giãn để hợp nhất các từ thành các dòng
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Tăng kích thước kernel để giãn ngang
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Tìm các contours của các dòng văn bản
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contours từ trên xuống dưới
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        line_img = img[y:y+h, x:x+w]
        line_image_path = os.path.join(output_dir, f'line_{i+1}.png')
        cv2.imwrite(line_image_path, line_img)
        
    images = []

    # Hàm khóa để sắp xếp các tệp theo số thứ tự
    def get_line_number(filename):
        match = re.search(r'line_(\d+)\.png', filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # Đảm bảo rằng các tệp không khớp sẽ được đặt cuối cùng

    # Liệt kê các tệp trong thư mục và sắp xếp theo số thứ tự
    image_files = sorted([f for f in os.listdir('tmp') if f.endswith('.png') or f.endswith('.jpg')], key=get_line_number)

    # Đọc từng ảnh và lưu vào mảng images
    for image_file in image_files:
        image_path = os.path.join('tmp', image_file)
        img = cv2.imread(image_path)
        images.append(img)
    
    prediction_text = ""
    for image in images:
        prediction_text = model.predict(image) + " "
        prediction_text = re.sub(r'\s+([.,:])', r'\1', prediction_text)
        prediction_text = re.sub(r'([.,:])\1+', r'\1', prediction_text)
        print(prediction_text)
            
    cv2.imshow("Image",img_cp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    shutil.rmtree('tmp')