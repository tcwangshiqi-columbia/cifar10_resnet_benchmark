A ResNet Benchmark on CIFAR-10 for Neural Network Verification
-----

We propose a new set of benchmarks of residual networks (ResNet) on CIFAR-10
for neural network verification in this repository.

Currently, most networks evaluated in the literature are feedforward NNs, and
many tools are hardcoded to handle feedforward networks only. To make neural
network verification more useful in practical scenarios, we advocate that tools
should handle more general architectures, and ResNet is the first step towards
this goal. We hope this can provide some incentives for the community to
develop better tools.

**Model details**: We provided two small ResNet models on CIFAR-10 with the following structures:

- ResNet-2B with 2 residual blocks: 5 convolutional layers + 2 linear layers
- ResNet-4B with 4 residual blocks: 9 convolutional layers + 2 linear layers

The ONNX format networks are available in the [onnx](onnx/) folder, and PyTorch models
can be found in the [pytorch_model](pytorch_model) folder. Model definitions
are available [here](pytorch_model/resnet.py).

Since this is one of the first benchmarks using ResNet, we keep the networks
relatively small (compared to ResNet 50 used in many vision tasks) so hopefully
most tools can handle them: the two networks have only 2 and 4 residual blocks,
respectively and they are relatively narrow -- although they are still large
compared to most models used in the complete NN verification literature. The
networks are trained using adversarial training with L_\infty perturbation
epsilon (2/255). We report basic model performance numbers below:

| Model      | # ReLUs | Clean acc. | PGD acc.(eps2/eps1) | CROWN/DeepPoly verified acc. |
|------------|---------|------------|---------------------|------------------------------|
| ResNet-2B  |   6244  |    69.25%  |    56.67%/63.15%    |   26.88%                     |
| ResNet-4B  |  14436  |    77.20%  |    62.92%/70.41%    |    0.24%                     |

Since the models are trained using adversarial training, it also poses a
challenge for many verifiers - the CROWN/DeepPoly verified accuracy (as a
simple baseline) is much lower than PGD accuracy, and we hope this benchmark
can motivate researchers in the community to develop stronger tools that can
make this gap smaller.

**Data Format**: The input images should be normalized using mean and std
computed from CIFAR-10 training set. The perturbation budget is element-wise,
eps=2/255 on unnormalized images and clipped to the [0, 1] range. We provide
`eval.py` as a simple PyTorch example of loading data (e.g., data
preprocessing, channel order etc).

**Data Selection**: We propose to randomly select 100 images from the test set
which are classified correctly and cannot be attacked by a 50-step targeted PGD
attack with 5 random restarts.  For each image, we specify the property that the groundtruth label
is always larger than all the other labels within L_\infty perturbation `eps=2/255` 
on input for resnet2b and `eps=1/255` for resnet4b. 

See instructions below for generating test images with a script, and some example
properties are in the `vnnlib_properties_pgd_filtered` folder.

To keep the sum of all the timeouts less than 6 hours, the suggested
per-example timeout is 5 minutes per example (assuming only a fraction of all
examples time out).


**Generating properties**: To generate properties from 100 random images
against runnerup labels (i.e., the property is that the true label is larger
than the runnerup label under perturbation) that are classified correctly and
are also robust against targeted pgd attacks, please run:

```bash
cd pytorch_model
python generate_properties_pgd.py --model resnet2b --num_images 100 --random True --epsilons '2/255'
python generate_properties_pgd.py --model resnet4b --num_images 100 --random True --epsilons '1/255'
```

To create the instance csv file for properties:
```bash
python create_instance.py
```
