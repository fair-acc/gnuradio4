#!/usr/bin/env python3
import tensorflow as tf
import tf2onnx

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    # Load the frozen graph
    with tf.compat.v1.Session() as sess:
        with tf.io.gfile.GFile("frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        # Use the names obtained from the frozen function printout
        input_name = "inputs:0"
        output_name = "Identity:0"

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            sess.graph,
            input_names=[input_name],
            output_names=[output_name],
            opset=15
        )

        model_proto = onnx_graph.make_model("peak_detector_model")
        with open("peak_detector.onnx", "wb") as fb:
            fb.write(model_proto.SerializeToString())

    print("Model exported to peak_detector.onnx")
