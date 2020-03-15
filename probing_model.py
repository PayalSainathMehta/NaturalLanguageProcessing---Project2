# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports
from util import load_pretrained_model
#import ipdb

class ProbingClassifier(models.Model):
    def __init__(self,
                 pretrained_model_path: str,
                 layer_num: int,
                 classes_num: int) -> 'ProbingClassifier':
        """
        It loads a pretrained main model. On the given input,
        it takes the representations it generates on certain layer
        and learns a linear classifier on top of these frozen
        features.

        Parameters
        ----------
        pretrained_model_path : ``str``
            Serialization directory of the main model which you
            want to probe at one of the layers.
        layer_num : ``int``
            Layer number of the pretrained model on which to learn
            a linear classifier probe.
        classes_num : ``int``
            Number of classes that the ProbingClassifier chooses from.
        """
        super(ProbingClassifier, self).__init__()
        self._pretrained_model = load_pretrained_model(pretrained_model_path)
        self._pretrained_model.trainable = False
        self._layer_num = layer_num

        # TODO(students): start
        # ...
        #Initialized a dense layer with softmax activation for probing.
        self.softmax_layer = tf.keras.layers.Dense(classes_num, activation = "softmax")
        # TODO(students): end

    def call(self, inputs: tf.Tensor, training: bool =False) -> tf.Tensor:
        """
        Forward pass of Probing Classifier.

        Parameters
        ----------
        inputs : ``str``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, max_tokens_num) and entries are indices of tokens
            in to the vocabulary. 0 means that it's a padding token. max_tokens_num
            is maximum number of tokens in any text sequence in this batch.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.
        """
        # TODO(students): start
        # ...
        #supplying the inputs to the pretrained model
        output = self._pretrained_model(inputs)
        #supplied number of layer to get the layer representations - returned as logits
        logits = self.softmax_layer(output['layer_representations'][:, self._layer_num-1, :])
        # TODO(students): end
        return {"logits": logits}
