#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import os

#Load Movie Reviews
from sklearn.datasets import load_files

#Download url: https://ai.stanford.edu/~amaas/data/sentiment/
reviews_train = load_files("D:/Dylan/Documents/MovieReviewDatabase/train")

text_train, y_train = reviews_train.data, reviews_train.target
print(f"type(text_train): {type(text_train)}")
print(f"Length: {len(text_train)}")
print(f"text_train[6]:\n{text_train[6]}")

#replace html formatting
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

#load test_data
reviews_test = load_files("D:/Dylan/Documents/MovieReviewDatabase/test")
text_test, y_test = reviews_test.data, reviews_test.target
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
