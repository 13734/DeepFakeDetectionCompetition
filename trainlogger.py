import os
import datetime


class TrainLogger:
    def __init__(self):
        self.DEFAULT_DICTIONARY:str = "./mylogger"
        self.FILE_NAME:str = "logger.txt"

        self.model_name :str= ""

        if not os.path.exists(self.DEFAULT_DICTIONARY):
            os.mkdir(self.DEFAULT_DICTIONARY)

    def writeText(self,add_string : str):
        with open(os.path.join(self.DEFAULT_DICTIONARY,self.FILE_NAME),"a",encoding='utf-8') as f:
            f.write(add_string)
            f.write("\n")

    def trainStart(self,model_name:str):
        self.model_name = model_name
        input_string : str = "\n\n[{}] model({})start".format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'),model_name)
        self.writeText(input_string)

    def writeTrainInfo(self,inform_dict : dict):
        input_string: str = ""
        for i_key in inform_dict.keys():
            input_string += "{}:{}\t".format(str(i_key),str(inform_dict[i_key]))
        input_string += datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        self.writeText(input_string)

if __name__ == '__main__':
    logger = TrainLogger()
    logger.trainStart("class_test")
    logger.writeTrainInfo({"epoch":1,"loss":0.9,"accurate":"(89.5)90.6"})

