from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")
model.set_classes(["person", "phone"])
results = model.predict("../my_test_image.jpg", imgsz=480)
print(results[0].boxes)
results[0].save("../test26_output.jpg")