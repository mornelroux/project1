#Standart Libraries
import os,sys
from datetime import datetime

#Global Paths
DATASETS_PATH = os.path.join(os.path.dirname(__file__), 'datasets')
RUNS_PATH = os.path.join(os.path.dirname(__file__), 'runs')
CATTLE_YAML_PATH = os.path.join(DATASETS_PATH, 'cattle','cattle.yaml')
SHEEP_YAML_PATH = os.path.join(DATASETS_PATH, 'sheep','sheep.yaml')

##### User Inputs ####################################################################
MODEL_PATH      = os.path.join(os.path.dirname(__file__), 'weights','pretrained','yolov10n.pt')
SESSION_NAME    = 'cattleTest'
DATASET         = CATTLE_YAML_PATH
############################################################################################

#Session variables
T_STAMP = datetime.now().strftime('%Y%m%d-%H:%M')
OUTPUT_PATH = os.path.join(RUNS_PATH, SESSION_NAME)
RUN_NAME = SESSION_NAME + '-' + T_STAMP

#Include and import Yolov10 from sub directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov10'))
from yolov10.ultralytics import YOLOv10

#Create YOLOv10 model
model = YOLOv10(MODEL_PATH)

#Train the model
model.train(
            data= DATASET,                  # Path to your dataset config file
            batch = 8,                      # Training batch size
            imgsz= 640,                     # Input image size
            epochs= 2,                      # Number of training epochs
            optimizer= 'SGD',               # Optimizer, can be 'Adam', 'SGD', etc.
            lr0= 0.01,                      # Initial learning rate
            lrf= 0.1,                       # Final learning rate factor
            weight_decay= 0.0005,           # Weight decay for regularization
            momentum= 0.937,                # Momentum (SGD-specific)
            verbose= True,                # Verbose output
            workers= 8,                   # Number of workers for data loading
            project= OUTPUT_PATH,       # Output directory for results
            name= RUN_NAME,                  # Experiment name
            exist_ok= False,              # Overwrite existing project/name directory
            rect= False,                  # Use rectangular training (speed optimization)
            resume= False,                # Resume training from the last checkpoint
            multi_scale= False,           # Use multi-scale training
            single_cls= False,             # Treat data as single-class
            scale = 0.75,
        )
