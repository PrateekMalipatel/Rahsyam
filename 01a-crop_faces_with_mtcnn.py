import cv2
from mtcnn import MTCNN
import sys, os.path
import json
import tensorflow as tf

print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# List available devices and determine which to use
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Check if any GPU (iGPU) is detected and enable memory growth if necessary
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Using iGPU:", physical_devices)
    # Enable memory growth on detected iGPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Using CPU as no iGPU detected.")

# Define base path
base_path = './train_sample_videos/'

# Helper function to get the file name without extension
def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

# Load metadata
metadata_path = os.path.join(base_path, 'metadata.json')
if os.path.exists(metadata_path):
    with open(metadata_path) as metadata_json:
        metadata = json.load(metadata_json)
        print(f"Metadata file contains {len(metadata)} items.")
else:
    print(f"Metadata file not found at {metadata_path}")
    sys.exit()

# Iterate over filenames in metadata
for filename in metadata.keys():
    tmp_path = os.path.join(base_path, filename[:-4])  # Removed extension
    print(f"Processing Directory: {tmp_path}")
    
    # Check if the directory exists
    if os.path.exists(tmp_path):
        frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
        faces_path = os.path.join(tmp_path, 'faces')
        os.makedirs(faces_path, exist_ok=True)
        print(f"Creating Directory: {faces_path}")
        print("Cropping faces from images...")

        # Initialize MTCNN detector once and reuse it
        detector = MTCNN()

        # Process each frame image
        for frame in frame_images:
            frame_path = os.path.join(tmp_path, frame)
            print(f"Processing {frame}")
            
            # Read and convert the image to RGB format
            image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            
            # Detect faces in the image
            results = detector.detect_faces(image)
            
            if not results:
                print(f"No faces detected in {frame}")
                continue
            
            print(f"Faces detected: {len(results)}")
            count = 0
            
            # Process each detected face
            for result in results:
                bounding_box = result['box']
                confidence = result['confidence']
                
                # Skip low confidence detections
                if confidence < 0.95:
                    print(f"Skipping low confidence detection (confidence: {confidence})")
                    continue
                
                # Calculate coordinates with margin
                margin_x = bounding_box[2] * 0.3  # 30% margin
                margin_y = bounding_box[3] * 0.3  # 30% margin
                x1 = max(0, int(bounding_box[0] - margin_x))
                x2 = min(image.shape[1], int(bounding_box[0] + bounding_box[2] + margin_x))
                y1 = max(0, int(bounding_box[1] - margin_y))
                y2 = min(image.shape[0], int(bounding_box[1] + bounding_box[3] + margin_y))
                
                print(f"Cropping face region: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Crop and save the face image
                crop_image = image[y1:y2, x1:x2]
                new_filename = os.path.join(faces_path, f"{get_filename_only(frame)}-{count:02d}.png")
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
                count += 1
    else:
        print(f"Directory not found: {tmp_path}")
