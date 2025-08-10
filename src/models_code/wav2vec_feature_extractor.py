from transformers import TFAutoModel
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class Wav2VecFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, model_checkpoint, **kwargs):
        super().__init__(**kwargs)
        self.model_checkpoint = model_checkpoint
        self.wav2vec = TFAutoModel.from_pretrained(model_checkpoint, trainable=False, from_pt=True)

    def call(self, inputs):
        outputs = self.wav2vec(inputs)
        return outputs.last_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_checkpoint": self.model_checkpoint,
        })
        return config