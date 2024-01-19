import tensorflow as tf
from tensorflow.keras.utils import plot_model


# モデルの定義
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# モデルの概要を出力
model.summary()

# モデルの図を画像として保存
plot_model(model, to_file="model.png", show_shapes=True)
