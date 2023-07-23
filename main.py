import cv2
import os
import pandas as pd
from deepface import DeepFace

data = {
    "Name": [],
    "Age": [],
    "Gender": [],
    "Race": [],
    "Emotion": []
}

for file in os.listdir("faces"):
    result = DeepFace.analyze(cv2.imread(f"faces/{file}"), actions=("age", "gender", "race", "emotion"))
    print(result)
    data["Name"].append(file.split(".")[0])
    data["Age"].append(result[0]["age"])
    data["Gender"].append(result[0]["dominant_gender"])
    data["Race"].append(result[0]["dominant_race"])
    data["Emotion"].append(result[0]["dominant_emotion"])


df = pd.DataFrame(data)
print(df)


df.to_csv("people.csv")