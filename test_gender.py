import os
import sys
import pandas as pd
import numpy as np

from audio_parser import get_frequencies, get_features
from joblib import load

def define_gender(wav_file_path: str) -> str:
    frequencies = get_frequencies(wav_file_path)
    if frequencies:
        features = np.array(get_features(frequencies)).reshape((1, -1))
        features = scaler.transform(features)
        y_pred = clf.predict(features)[0]
        return "male" if y_pred == 1 else "female"
    raise Exception("Wrong wav-file format")

def main():
    _, inp, *_ = sys.argv
    if os.path.isdir(inp):
        for wav_name in os.listdir(inp):
            gender = define_gender(os.path.join(inp, wav_name))
            print(f"{wav_name}->{gender}")
    else:
        gender = define_gender(inp)
        print(f"{os.path.basename(inp)}->{gender}")

if __name__ == "__main__":
    clf = load("best_model.joblib")
    scaler = load("scaler.joblib")
    main()
