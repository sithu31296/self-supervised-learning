# Self Supervised Image Classification

This is the task of image classification using representations learnt with self-supervised learning. Self-supervised methods generally involve a pretext task that is solved to learn a good representation and a loss function to learn with. One example of a loss function is an autoencoder based loss where the goal is reconstruction of an image pixel-by-pixel. A more popular recent example is a contrastive loss, which measure the similarity of sample pairs in a representation space, and where there can be a varying target instead of a fixed target to reconstruct (as in the case of autoencoders).

A common evaluation protocol is to train a linear classifier on top of (frozen) representations learnt by self-supervised methods. The leaderboards for the linear evaluation protocol can be found below. In practice, it is more common to fine-tune features on a downstream task. An alternative evaluation protocol therefore uses semi-supervised learning and finetunes on a % of the labels. 

https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html

https://amitness.com/2020/02/illustrated-self-supervised-learning/

https://www.fast.ai/2020/01/13/self_supervised/

https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html

https://vimeo.com/390347111




## Model Comparison

Model | Backbone | ImageNet Top-1 Accuracy | Params (M)
--- | --- | --- | ---
EsViT | Swin-B/14 | 81.3 | 87
EsViT | Swin-S/14 | 80.8 | 49
EsViT | Swin-T/14 | 78.7 | 28
DINO | XCiT-M24/8 | 80.3 | 84
DINO | XCiT-S12/8 | 79.2 | 26
DINO | ViT-B/8 | 80.1 | 85
DINO | ViT-S/8 | 79.7 | 21