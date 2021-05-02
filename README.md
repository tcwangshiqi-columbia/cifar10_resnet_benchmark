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

ResNet-2B with 2 residual block: 5 convolutional layers + 2 linear layers
ResNet-4B with 4 residual blocks: 9 convolutional layers + 2 linear layers

Since this is the first benchmark using ResNet, we keep the networks relatively
small so hopefully most tools can handle them: the two networks have only 2 and
4 residual blocks, respectively and they are relatively narrow. The networks
are trained using adversarial training with L_\infty perturbation epsilon
(2/255). We report basic model performance numbers below:

| Model      | Clean acc. | PGD acc. | CROWN/DeepPoly verified acc. |
|------------|------------|----------|------------------------------|
| ResNet-2B  |    69.25%  |  56.67%  |   26.88%                     |
| ResNet-4B  |    67.21%  |  55.25%  |    8.81%                     |

Since the models are trained using adversarial training, it also poses a
challenge for many verifiers - the CROWN/DeepPoly verified accuracy is much
lower than PGD accuracy, and we hope this benchmark can motivate researchers in
the community to develop stronger tools that can make this gap smaller.

**Data Selection**: we proposed to randomly select 100 images from the test set
which are classified correctly and cannot be attacked by 100-step PGD. For each
image, we currently specify all 9 other labels as the target labels for
verification, and verify the property that the no label is not larger than the
groundtruth label within L_\infty `eps=2/255` on input. To keep the sum of all
the timeouts less than 6 hours (as @stanleybak suggested), the suggested
per-example timeout is 6 minutes (assuming only a fraction of all examples time
out).

