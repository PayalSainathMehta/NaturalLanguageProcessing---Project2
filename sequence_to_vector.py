# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
import ipdb


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor, #original input words
             sequence_mask: tf.Tensor, #padding into words
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.dropout = dropout
        #Initializing a dense layer with an activation function of Relu (Rectified Linear Unit)
        self.input_layers = [tf.keras.layers.Dense(input_dim,activation = "relu") for i in range(num_layers)]
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        
        # TODO(students): start
        # ...
        sequence_mask = tf.cast(sequence_mask,tf.float32)
        length = vector_sequence.get_shape().as_list()[0]
        #ipdb.set_trace()
        num_tokens = vector_sequence.get_shape().as_list()[1]
        threshold = 5
        #Checking below condition to avoid NaNs in lesser number of tokens in input sequence
        if training and num_tokens > threshold:
        	#Calculating random probabilites
            prob_distribution = tf.random.uniform(sequence_mask.shape)
            #Truth values
            #ipdb.set_trace()
            #Dropping random probabilites lesser than dropout probability
            prob_distribution = prob_distribution > self.dropout
            prob_distribution = tf.cast(prob_distribution,tf.float32)
            #ipdb.set_trace()
            #new sequence mask
            sequence_mask = sequence_mask * prob_distribution
        #Increasing dimensions of the sequence mask so as to multiply with the input vector
        sequence_mask_3D = tf.tile(sequence_mask, [1, 50])
        sequence_mask_3D = tf.reshape(sequence_mask_3D, [1, -1])
        sequence_mask_3D = tf.reshape(sequence_mask_3D, [length, 50, -1])
        sequence_mask_3D = tf.transpose(sequence_mask_3D, [0, 2, 1])
        #Multiplying the input vector with the sequence mask
        dropped_vectors = vector_sequence * sequence_mask_3D
        numerator = tf.math.reduce_sum(dropped_vectors, 1)
        denominator = tf.math.reduce_sum(sequence_mask, 1)
        denominator = tf.reshape(denominator, [-1,1])
        #Doing below to avoid NaNs because of dropout probability being high or input being a bigramn model. 
        #To avoid division by 0.
        denominator = tf.maximum(denominator, 1)
        result_average = tf.math.divide(numerator, denominator)
        temp_list = [result_average]
        for i in range(len(self.input_layers)):
            output = self.input_layers[i](temp_list[i])
            temp_list.append(output)
        combined_vector = temp_list[len(temp_list)-1]
        layer_representations = tf.transpose(tf.stack(temp_list),[1,0,2])
        #ipdb.set_trace()
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        #Initializing the GRU layer
        self.input_layers_GRU = [tf.keras.layers.GRU(input_dim, return_sequences = True) for i in range(num_layers)]
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...
        temp_list = []
        output = vector_sequence
        for i in range(len(self.input_layers_GRU)):
            #we pass the input vector to the GRU first layer
            output = self.input_layers_GRU[i](output, mask = sequence_mask)
            #we now slice and take only the 64*50 vectors from the output and append it to a list
            output_2D = output[:, -1, :]
            temp_list.append(output_2D)
        #return combined vector as the output of the last layer and layer representations as the sum of outputs of all layers.
        combined_vector = temp_list[-1]
        layer_representations = tf.transpose(tf.stack(temp_list),[1,0,2])


        #ipdb.set_trace()
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
