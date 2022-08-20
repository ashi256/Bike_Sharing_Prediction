import os
import sys

from sharing.exception import SharingException
from sharing.util.util import load_object

import pandas as pd


class SharingData:

    def __init__(self,         
                season: str ,       
                yr: str      ,       
                mnth: str     ,      
                hr: str       ,      
                holiday:str  ,      
                weekday: str  ,      
                
                weathersit: str ,    
                temp : float ,         
                     
                windspeed: float ,   
                casual: int  ,       
                registered : int,    
                cnt: int =None,
                 ):
        try:
            self.season = season
            self.yr = yr
            self.mnth = mnth
            self.hr = hr
            self.holiday = holiday
            self.weekday = weekday
            
            self.weathersit = weathersit
            self.temp = temp
          
            self.windspeed = windspeed
            self.casual = casual
            self.registered = registered  
            self.cnt = cnt

        except Exception as e:
            raise SharingException(e, sys) from e

    def get_sharing_input_data_frame(self):

        try:
            sharing_input_dict = self.get_sharing_data_as_dict()
            return pd.DataFrame(sharing_input_dict)
        except Exception as e:
            raise SharingException(e, sys) from e

    def get_sharing_data_as_dict(self):
        try:
            input_data = {         
                "season": [self.season] ,       
                "yr": [self.yr]      ,       
                "mnth": [self.mnth]     ,      
                "hr": [self.hr]       ,      
                "holiday": [self.holiday]  ,      
                "weekday": [self.weekday]  ,      
              
                "weathersit": [self.weathersit] ,    
                "temp" : [self.temp] ,         
                     
                "windspeed": [self.windspeed] ,   
                "casual": [self.casual]  ,       
                "registered" : [self.registered]
                }
            return input_data
        except Exception as e:
            raise SharingException(e, sys)


class SharingPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise SharingException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise SharingException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            count = model.predict(X)
            return count
        except Exception as e:
            raise SharingException(e, sys) from e