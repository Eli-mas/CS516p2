# CS516p2

Convolutional neural networks for classifying facial data for William & Mary's CS 516 (Intro to Machine Learning). Included functionality:
* Train, save, evaluate neural networks producing single or multi-class labels using [tensorflow.keras](https://www.tensorflow.org/api_docs/python/tf/keras)
* Automatically copy data between local and remote machines via [scp](https://en.wikipedia.org/wiki/Secure_copy_protocol),<a href="https://en.wikipedia.org/wiki/SSH_(Secure_Shell)">ssh</a>
* Monitor the state of GPU devices on remote host via tensorflow's [configuration interface](https://www.tensorflow.org/api_docs/python/tf/config)