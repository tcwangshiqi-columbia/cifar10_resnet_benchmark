import sys
import os
import csv

model_name = sys.argv[1]

assert os.path.exists(f"onnx/{model_name}.onnx")
assert os.path.exists("vnnlib_properties/")

name = ["model_name", "property_name", "timeout"]
instance_list = []
for i in range(100):
	instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties/prop_{i}_eps_0.008.vnnlib", "360"])

with open(model_name+'_instance.csv', 'w') as f:
    write = csv.writer(f)
    # write.writerow(fields)
    write.writerows(instance_list)

