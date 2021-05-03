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

Since this is one of the first benchmarks using ResNet, we keep the networks
relatively small (compared to ResNet 50 used in many vision tasks) so hopefully
most tools can handle them: the two networks have only 2 and 4 residual blocks,
respectively and they are relatively narrow -- although they are still large
compared to most models used in the complete NN verification literature. The
networks are trained using adversarial training with L_\infty perturbation
epsilon (2/255). We report basic model performance numbers below:

| Model      | Clean acc. | PGD acc. | CROWN/DeepPoly verified acc. |
|------------|------------|----------|------------------------------|
| ResNet-2B  |    69.25%  |  56.67%  |   26.88%                     |
| ResNet-4B  |    67.21%  |  55.25%  |    8.81%                     |

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
attack with 5 random restarts.  For each image, we specify the runner up label
(the class with second largest logit) as the target for verification, and
verify the property that the logit of runner up label is not larger than that
of the groundtruth label within L_\infty perturbation `eps=2/255` on input. See
instructions below for generating test images with a script, and some example
properties are in the `vnnlib_properties_pgd_filtered` folder.

To keep the sum of all the timeouts less than 6 hours, the suggested
per-example timeout is 6 minutes per example (assuming only a fraction of all
examples time out).


**Generating properties**: To generate properties from 100 random images
against runnerup labels (i.e., the property is that the true label is larger
than the runnerup label under perturbation) that are classified correctly and
are also robust against targeted pgd attacks, please run:

```bash
cd pytorch_model
python generate_properties_pgd.py --model resnet2b --num_images 100 --random True --epsilons '2/255'
python generate_properties_pgd.py --model resnet4b --num_images 100 --random True --epsilons '2/255'
```

To generate the properties from 100 random images against all 9 other target
labels (i.e., the property is that the true label is larger than *all* other 9
labels under perturbation), please run:

```bash
python generate_properties.py --num_images 100 --random True --epsilons '2/255'
```
This setting is useful for computing a "verified accuracy" given a dataset.

