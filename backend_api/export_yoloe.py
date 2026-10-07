from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")
model.set_classes(["person", "mobile phone", "spectacles"])
model.export(format="onnx", imgsz=(224, 320), opset=17, simplify=True, end2end=False)