import os

dataset_path = r"C:\Users\WIN\OneDrive\Desktop\DeepFake_Detection\Dataset"

splits = ["Train", "Validation", "Test"]
categories = ["Real", "Fake"]

for split in splits:
    print(f"\n--- {split} Dataset ---")

    for category in categories:
        folder = os.path.join(dataset_path, split, category)

        if not os.path.isdir(folder):
            print(f"{category}: Folder NOT FOUND")
            continue

        count = sum(
            1 for file in os.listdir(folder)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".bmp")
            )
        )

        print(f"{category}: {count} images")