
from src.preprocessing.histogram import (
    load_grayscale,
    calculate_local_anomalies,
)

image = load_grayscale(r"C:\Users\Siddhartha\Desktop\EchoTrace\data\KLSG\archive\valid\images\000010_jpg.rf.efb6cf8c3257d078e4fffc1451283b5b.jpg")

anomalies = calculate_local_anomalies(image)

for region in anomalies:
    print(region)
