import tensorflow as tf
from rnn import train_and_save_model
from config import new_rnn_path


class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)


train_and_save_model()


inputs = ['как поживать диплом?) уже создать папка на рабочий стол?',
         'это быть круто!!!)))) спасибо всем!!! всем, кто',
         'есть вероятность того, что сегодня быть принималово,']


reloaded = tf.saved_model.load(new_rnn_path)
_ = reloaded.translate(tf.constant(inputs))  # warmup
result = reloaded.translate(tf.constant(inputs))

print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()
