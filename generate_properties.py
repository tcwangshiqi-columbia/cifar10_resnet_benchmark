import os
os.system("python pytorch/generate_properties_pgd.py --model resnet2b --num_images 50 --random True --epsilons '2/255'")
os.system("python pytorch/generate_properties_pgd.py --model resnet4b --num_images 50 --random True --epsilons '1/255'")
os.system("python create_instance.py")