from ultralytics import YOLO
import cv2
import datetime
from collections import defaultdict
import random
import psycopg2

model = YOLO("yolov10s.pt")

video_path = "Sample Object Detection Video.mp4"
cap = cv2.VideoCapture(video_path)

# Setup for saving output video
output_path = "Output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

detected_data = []
object_counts = defaultdict(int)
class_colors = {}

# PostgreSQL connection setup
try:
    conn = psycopg2.connect(
        dbname="OBJECT DETECTION AND COUNTING",
        user="postgres",
        password="682037",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    print("Connected to PostgreSQL database.")
except Exception as e:
    print("Failed to connect to PostgreSQL:", e)
    exit()

def get_color(class_name):
    if class_name not in class_colors:
        random.seed(hash(class_name) % 2**32)
        class_colors[class_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return class_colors[class_name]

print("STARTING Video Object Detection... Press 'q' to Quit.")

# Main detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete.")
        break

    results = model(frame, save=False)[0]
    current_time = datetime.datetime.now()

    if not results.boxes:
        print("No detections in this frame.")
        out_writer.write(frame)  # still save frame without detections
        continue

    for box in results.boxes.data.tolist():
        class_id = int(box[5])
        class_name = model.names[class_id]
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')

        detected_data.append({"class_name": class_name, "timestamp": timestamp})
        object_counts[class_name] += 1

        try:
            cursor.execute(
                'INSERT INTO "OBJECT DETECTION" ("Object", "Timestamp") VALUES (%s, %s)',
                (class_name, timestamp)
            )
            conn.commit()
        except Exception as db_error:
            print("Error inserting into PostgreSQL:", db_error)

        x1, y1, x2, y2 = map(int, box[:4])
        color = get_color(class_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv10 Video Detection", frame)
    out_writer.write(frame)  # === NEW: Save annotated frame to output video ===

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Detection stopped by user.")
        break

cap.release()
out_writer.release()  # === NEW ===
cv2.destroyAllWindows()
cursor.close()
conn.close()

# Save to CSV file
with open("detections.csv", "w") as f:
    f.write(f"{'class_name'.ljust(20)}{'timestamp'}\n")
    for entry in detected_data:
        f.write(f"{entry['class_name'].ljust(20)}{entry['timestamp']}\n")

# Print count output
print("Detection Data Saved to detections.csv")
print("\nTotal Object Counts:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")