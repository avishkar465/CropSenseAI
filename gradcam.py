import numpy as np


class GradCAMError(RuntimeError):
    pass


def generate_gradcam(model, image_array, class_index: int) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap for a compatible Keras CNN."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise GradCAMError("TensorFlow is unavailable, so Grad-CAM cannot be generated.") from exc

    input_batch = tf.convert_to_tensor(np.asarray(image_array, dtype=np.float32)[None, ...])
    target_model = None
    target_model_index = -1
    try:
        model_layers = model.layers
    except AttributeError as exc:
        raise GradCAMError("This model does not expose a supported convolutional layer.") from exc
    for layer_index in range(len(model_layers) - 1, -1, -1):
        layer = model_layers[layer_index]
        if isinstance(layer, tf.keras.Model):
            convolutional_layers = [
                candidate for candidate in layer.layers if isinstance(candidate, tf.keras.layers.Conv2D)
            ]
            if convolutional_layers:
                target_model = layer
                target_model_index = layer_index
                target_layer = convolutional_layers[-1]
                break

    try:
        if target_model is not None:
            activation_model = tf.keras.Model(
                target_model.input, [target_layer.output, target_model.output]
            )
            with tf.GradientTape() as tape:
                backbone_input = input_batch
                for layer in model_layers[1:target_model_index]:
                    backbone_input = layer(backbone_input, training=False)
                conv_outputs, predictions = activation_model(backbone_input, training=False)
                for layer in model_layers[target_model_index + 1:]:
                    predictions = layer(predictions, training=False)
                class_score = predictions[:, class_index]
        else:
            convolutional_layers = [
                layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
            ]
            if not convolutional_layers:
                raise GradCAMError("This model does not expose a supported convolutional layer.")
            target_layer = convolutional_layers[-1]
            grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_batch, training=False)
                class_score = predictions[:, class_index]

        gradients = tape.gradient(class_score, conv_outputs)
        if gradients is None:
            raise GradCAMError("Could not connect the selected layer to the model prediction.")
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0)
        maximum = tf.reduce_max(heatmap)
        heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
        return np.asarray(heatmap, dtype=np.float32)
    except GradCAMError:
        raise
    except Exception as exc:
        raise GradCAMError("Grad-CAM could not be computed for this model architecture.") from exc


def create_overlay(image_path, heatmap, output_path, alpha: float = 0.38) -> None:
    """Blend a real Grad-CAM heatmap over its source image and save the result."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        from PIL import Image

        with Image.open(image_path) as source:
            base = source.convert("RGB")
            resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(base.size)
            color = Image.fromarray(np.uint8(cm.get_cmap("turbo")(np.asarray(resized) / 255)[..., :3] * 255))
            Image.blend(base, color, alpha).save(output_path, format="JPEG", quality=90)
    except Exception as exc:
        raise GradCAMError("The Grad-CAM overlay could not be saved.") from exc