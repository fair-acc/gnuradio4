# freeze_tf2.py
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

saved_model_dir = "saved_model_dir"

# Load SavedModel in TF2 eager mode
saved_model = tf.saved_model.load(saved_model_dir)
f = saved_model.signatures['serving_default']

# Convert to a frozen function
frozen_func = convert_variables_to_constants_v2(f, lower_control_flow=False)
graph_def = frozen_func.graph.as_graph_def()

# Print the actual input/output tensor names:
print("Frozen function inputs:")
for inp in frozen_func.inputs:
    print("  ", inp.name)
print("Frozen function outputs:")
for out in frozen_func.outputs:
    print("  ", out.name)

# Save the frozen graph
with open("frozen_graph.pb", "wb") as fb:
    fb.write(graph_def.SerializeToString())
print("Frozen graph saved to frozen_graph.pb")
