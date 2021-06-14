import sys
import os
import csv

name = ["model_name", "property_name", "timeout"]
instance_list = []

# 48 properties for resnet2b
model_name = "resnet_2b"
assert os.path.exists(f"onnx/{model_name}.onnx")
assert os.path.exists("vnnlib_properties_pgd_filtered/")
for i in range(48):
    instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_{i}_eps_0.008.vnnlib", "300"])

# 24 properties for resnet2b
model_name = "resnet4b_wide"
assert os.path.exists(f"onnx/{model_name}.onnx")
for i in range(24):
    instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet4b_wide_pgd_filtered_1_255/prop_{i}_eps_0.004.vnnlib", "300"])

with open('instance.csv', 'w') as f:
    write = csv.writer(f)
    # write.writerow(fields)
    write.writerows(instance_list)

