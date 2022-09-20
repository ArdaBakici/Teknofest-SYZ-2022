<img align="left" src="https://upload.wikimedia.org/wikipedia/tr/1/1d/Teknofest_logo.png">  

# **DiagnosisByAI**
# *Teknofest 2022 Winner*
Codes of the team ***DiagnosisByAI*** winner of Teknofest 2022 Artifical Intelligence in Health Competition Disease Detection with Computer Vision Category High School Level

<br>

## Model Architecture

## Contents
manager.ipynb -> Main Python notebook  
converter.ipynb -> Python notebook for converting DICOM to JPG  
detect1.py -> Acute Appendicitis Detecting Model Codes  
detect2.py -> Acute Cholecystitis Detecting Model Codes  
detect3.py -> Acute Pancreatitis Detecting Model Codes  
detect4.py -> Kidney and Bladder Stone Detecting Model Codes  
detect5.py -> Acute Diverticulitis Detecting Model Codes  
detect6.py -> Acute Aortic Aneurysm and Dissection Detecting Model Codes  
/weights -> put model weights here (weights should have the same names with example txt files)  
/utils -> Scaled YOLOv4 util codes (taken from ![Scaled YOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4))  
/models -> Scaled YOLOv4 model codes (taken from ![Scaled YOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4))  
/dicom2jpg -> library for converting dicom to jpg (taken from ![dicom2jpg](https://github.com/ykuo2/dicom2jpg) with minor modifications)  
/output -> folder where results are saved

## How to Use ?
1- Put your dicom files inside a input folder  
2- Run converter.ipynb on your input folder  
3- Run manager.ipynb on output folder of converter.ipynb  
4- Wait...  
5- Your results are ready
