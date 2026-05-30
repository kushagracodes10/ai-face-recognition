import cv2
import os

folder = "known_faces"

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(folder, filename)
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read {filename}")
                continue
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Save back as proper 8-bit JPG
            cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            print(f"Converted {filename} to proper 8-bit RGB")
        except Exception as e:
            print(f"Error converting {filename}: {e}")
