import os

dataset_path = r"C:\Users\WIN\OneDrive\Desktop\DeepFake_Detection\Dataset"

folders = [
    "Train/Real",
    "Train/Fake",
    "Validation/Real",
    "Validation/Fake",
    "Test/Real",
    "Test/Fake"
]

print("Checking DeepFake dataset...\n")

for folder in folders:
    path = os.path.join(dataset_path, folder)

    if os.path.isdir(path):
        print(f"FOUND: {folder}")
    else:
        print(f"NOT FOUND: {folder}")

print("\nDataset verification complete!")