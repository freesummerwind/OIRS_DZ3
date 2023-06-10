import einops
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text

from config import rnn_dataset_path, new_rnn_path


def prepare_text(text, max_len=100):
    if len(text) < 20:
        return ''
    if len(text) < max_len:
        return text
    texts = text.split('\n')
    texts = [x.split('. ') for x in texts]
    texts = [x for arr in texts for x in arr]
    lens = np.array([len(x) for x in texts])
    lens_cum_sum = np.cumsum(lens)
    if lens_cum_sum[0] > max_len:
        texts = [x.split(' ') for x in texts]
        texts = [x for arr in texts for x in arr]
        lens = np.array([len(x) for x in texts])
        lens_cum_sum = np.cumsum(lens)
    max_count = len(lens_cum_sum[np.where(lens_cum_sum < max_len)[0]])
    texts = ' '.join(texts[:max_count])
    return texts


def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
#   text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-zа-я.?!,()]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿()/]', r'  ')
    # Strip whitespace.
    text = tf.strings.regex_replace(text, ' +', r' ')
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


# @title
class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                                   mask_zero=True)

        # The RNN layer processes those vectors sequentially.
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(units,
                                      # Return the sequence and state
                                      return_sequences=True,
                                      recurrent_initializer='glorot_uniform'))

    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        # 2. The embedding layer looks up the embedding vector for each token.
        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        # 3. The GRU processes the sequence of embeddings.
        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        # 4. Returns the new sequence of embeddings.
        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        shape_checker = ShapeChecker()

        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')

        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)

        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        # Cache the attention scores for plotting later.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units


        # 1. The embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   units, mask_zero=True)

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # 3. The RNN output will be the query for the attention layer.
        self.attention = CrossAttention(units)

        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)


@Decoder.add_method
def call(self,
         context, x,
         state=None,
         return_state=False):
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t')
    shape_checker(context, 'batch s units')

    # 1. Lookup the embeddings
    x = self.embedding(x)
    shape_checker(x, 'batch t units')

    # 2. Process the target sequence.
    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'batch t units')

    # 3. Use the RNN output as the query for the attention over the context.
    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'batch t units')
    shape_checker(self.last_attention_weights, 'batch t s')

    # Step 4. Generate logit predictions for the next token.
    logits = self.output_layer(x)
    shape_checker(logits, 'batch t target_vocab_size')

    if return_state:
        return logits, state
    else:
        return logits


@Decoder.add_method
def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result


@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature=0.0):
    logits, state = self(
        context, next_token,
        state=state,
        return_state=True)

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :] / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state


class Translator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units,
               context_text_processor,
               target_text_processor):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        #TODO(b/250038731): remove this
        try:
          # Delete the keras mask, so keras doesn't scale the loss+accuracy.
          del logits._keras_mask
        except AttributeError:
          pass

        return logits


def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask)


# @title
@Translator.add_method
def translate(self,
              texts, *,
              max_length=50,
              temperature=0.0):
    # Process the input texts
    context = self.encoder.convert_input(texts)
    batch_size = tf.shape(texts)[0]

    tokens = []
    attention_weights = []
    next_token, done, state = self.decoder.get_initial_state(context)

    for _ in range(max_length):
        # Generate the next token
        next_token, done, state = self.decoder.get_next_token(
            context, next_token, done, state, temperature)

        # Collect the generated tokens
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)

        if tf.executing_eagerly() and tf.reduce_all(done):
            break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)  # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

    result = self.decoder.tokens_to_text(tokens)
    return result


class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)


def train_and_save_model():
    df = pd.read_csv(rnn_dataset_path)
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
    df['source'] = [prepare_text(x, 100) for x in df['source']]
    df['target'] = [prepare_text(x, 100) for x in df['target']]
    df = df[df['source'] != '']
    df = df[df['target'] != '']
    print(f'Dirty df shape: {df.shape}')
    # print(df.head())
    # pattern = "[a-zA-Z]+"
    # filter = df['source'].str.contains(pattern)
    # df = df[~filter]
    # filter = df['target'].str.contains(pattern)
    # df = df[~filter]
    # df = df.dropna()
    # print(f'Clean df shape: {df.shape}')
    print(df.head())

    context_raw, target_raw = df.source.values, df.target.values
    context_raw = np.array([x.lower() for x in context_raw])
    target_raw = np.array([x.lower() for x in target_raw])

    BUFFER_SIZE = len(context_raw)
    BATCH_SIZE = 64

    is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE))
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE))

    for example_context_strings, example_target_strings in train_raw.take(1):
        print(example_context_strings[:5])
        print()
        print(example_target_strings[:5])
        break

    max_vocab_size = 14000

    context_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size,
        ragged=True)

    context_text_processor.adapt(train_raw.map(lambda context, target: context))
    print(context_text_processor.get_vocabulary()[:10])

    target_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size,
        ragged=True)
    target_text_processor.adapt(train_raw.map(lambda context, target: target))
    print(target_text_processor.get_vocabulary()[:10])

    def process_text(context, target):
        context = context_text_processor(context).to_tensor()
        target = target_text_processor(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()
        return (context, targ_in), targ_out

    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    print('OUTPUT AFTER DS')
    for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
        print(ex_context_tok[0, :10].numpy())
        print()
        print(ex_tar_in[0, :10].numpy())
        print(ex_tar_out[0, :10].numpy())

    UNITS = 256

    encoder = Encoder(context_text_processor, UNITS)
    ex_context = encoder(ex_context_tok)
    print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
    print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')

    attention_layer = CrossAttention(UNITS)

    # Attend to the encoded tokens
    embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(),
                                      output_dim=UNITS, mask_zero=True)
    ex_tar_embed = embed(ex_tar_in)
    result = attention_layer(ex_tar_embed, ex_context)

    print(f'Context sequence, shape (batch, s, units): {ex_context.shape}')
    print(f'Target sequence, shape (batch, t, units): {ex_tar_embed.shape}')
    print(f'Attention result, shape (batch, t, units): {result.shape}')
    print(f'Attention weights, shape (batch, t, s):    {attention_layer.last_attention_weights.shape}')

    print(attention_layer.last_attention_weights[0].numpy().sum(axis=-1))

    decoder = Decoder(target_text_processor, UNITS)
    logits = decoder(ex_context, ex_tar_in)

    print(f'encoder output shape: (batch, s, units) {ex_context.shape}')
    print(f'input target tokens shape: (batch, t) {ex_tar_in.shape}')
    print(f'logits shape shape: (batch, target_vocabulary_size) {logits.shape}')

    # Setup the loop variables.
    next_token, done, state = decoder.get_initial_state(ex_context)
    tokens = []

    for n in range(10):
        # Run one step.
        next_token, done, state = decoder.get_next_token(
            ex_context, next_token, done, state, temperature=1.0)
        # Add the token to the output.
        tokens.append(next_token)

    # Stack all the tokens together.
    tokens = tf.concat(tokens, axis=-1)  # (batch, t)

    # Convert the tokens back to aa string
    result = decoder.tokens_to_text(tokens)
    result[:3].numpy()

    model = Translator(UNITS, context_text_processor, target_text_processor)
    logits = model((ex_context_tok, ex_tar_in))

    print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
    print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
    print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')

    model.compile(optimizer='adam',
                  loss=masked_loss,
                  metrics=[masked_acc, masked_loss])

    vocab_size = 1.0 * target_text_processor.vocabulary_size()
    {"expected_loss": tf.math.log(vocab_size).numpy(),
     "expected_acc": 1 / vocab_size}

    print(model.evaluate(val_ds, steps=20, return_dict=True))

    history = model.fit(
        train_ds.repeat(),
        epochs=10,
        steps_per_epoch=100,
        validation_data=val_ds,
        validation_steps=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3)])

    result = model.translate(['чего блин происходит непонятно ахах а хотелось бы лето'])
    print(result[0].numpy().decode())

    inputs = [
        'Как поживают дипломы?) Уже создали папку на рабочем столе?',
        'Это было круто!!!)))) спасибо всем!!! Всем, кто помогал, ректору( рад был познакомиться)))!!!',
        'Есть вероятность того, что сегодня будет принималово, вероятность стремится к бесконечности)'
    ]
    inputs = [prepare_text(x, 100).lower() for x in inputs]
    result = model.translate(inputs)

    print(result[0].numpy().decode())
    print(result[1].numpy().decode())
    print(result[2].numpy().decode())
    print()

    export = Export(model)
    result = export.translate(inputs)

    tf.saved_model.save(export, new_rnn_path,
                        signatures={'serving_default': export.translate})

    reloaded = tf.saved_model.load(new_rnn_path)
    _ = reloaded.translate(tf.constant(inputs))
    result = reloaded.translate(tf.constant(inputs))
    print(result[0].numpy().decode())
    print(result[1].numpy().decode())
    print(result[2].numpy().decode())
    print()


# train_and_save_model()

