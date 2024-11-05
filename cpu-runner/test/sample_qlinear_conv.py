##
##  © Copyright 2022 – 2023 Advanced Micro Devices, Inc. All rights reserved.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##

# %%
import pathlib

HOME = pathlib.Path.home()
work_dir = HOME / "build" / "vaip_regression" / "qconv"
import xir
import numpy as np


def random_randint(shape):
    return random.randint(-(2 ** (7 - 1)), 2 ** (7 - 1) - 1, size=shape)


def create_cpu_sg(root, ops):
    cpu_sg = root.merge_children(set(root.find_child_subgraph_by_op(op) for op in ops))
    cpu_sg.set_attr("device", "CPU")


def create_user_sg(root, op_input):
    user_sg = root.find_child_subgraph_by_op(op_input)
    user_sg.set_attr("device", "USER")


def create_const_fix_op(graph, name, shape, type, data):
    return graph.create_op(
        name,
        "const-fix",
        {"shape": shape, "data_type": type, "data": data},
    )


def create_const_op(graph, name, shape, type, data):
    return graph.create_op(
        name,
        "const",
        {"shape": shape, "data_type": type, "data": data},
    )


def create_dequantize_op(graph, name, type1, data_type, i, s, z):
    return graph.create_op(
        name,
        type1,
        {"data_type": data_type},
        {"input": [i], "scale": [s], "zero_point": [z]},
    )


def create_qconv_xmodel(input_shape, weight_shape, bias_shape, weight_data, bias_data):
    g = xir.Graph("sample_qlinear_conv")
    op_input = g.create_op(
        "input", "data-fix", {"shape": input_shape, "data_type": "XINT8"}
    )
    op_weight = create_const_op(g, "weight1", weight_shape, "INT32", weight_data)
    op_bias = create_const_op(g, "bias1", bias_shape, "INT32", bias_data)

    op_qconv = g.create_op(
        "qconv1",
        "qlinear-conv2d",
        {
            "auto_pad": "SAME",
            "pad": [1, 1, 1, 1],
            "kernel": [3, 3],
            "stride": [1, 1],
            "x_scale": [1],
            "x_zero_point": [0],
            "w_scale": [1],
            "w_zero_point": [0],
            "y_scale": [100],
            "y_zero_point": [0],
        },
        {"input": [op_input], "weights": [op_weight], "bias": [op_bias]},
    )

    root = g.get_root_subgraph()
    root.create_child_subgraph()
    create_user_sg(root, op_input)
    create_cpu_sg(root, [op_weight, op_bias, op_qconv])
    g.serialize(str(work_dir / "a.xmodel"))
    print(str(work_dir / "a.xmodel"))


def create_dequantize_conv_xmodel(
    input_shape, weight_shape, bias_shape, weight_data, bias_data
):
    g = xir.Graph("sample_qlinear_conv")
    op_input = g.create_op(
        "input", "data-fix", {"shape": input_shape, "data_type": "XINT8"}
    )
    ops = []
    zero = np.array([0], dtype=np.int32)
    print("HELLLO")
    ops.append(
        create_const_op(g, "input_sc", [1], "FLOAT32", np.array([1], dtype=np.float32))
    )  # 0
    ops.append(create_const_op(g, "input_z", [1], "INT32", zero))  # 1
    ops.append(
        create_dequantize_op(
            g,
            "dequantize-input",
            "dequantize-linear",
            "FLOAT32",
            op_input,
            ops[0],
            ops[1],
        )
    )  # 2
    ops.append(create_const_op(g, "weight1", weight_shape, "INT32", weight_data))  # 3
    ops.append(
        create_const_op(g, "weight_s", [1], "FLOAT32", np.array([1], dtype=np.float32))
    )  # 4
    ops.append(create_const_op(g, "weight_z", [1], "INT32", zero))  # 5
    ops.append(
        create_dequantize_op(
            g,
            "dequantize-weight1",
            "dequantize-linear",
            "FLOAT32",
            ops[3],
            ops[4],
            ops[5],
        )
    )  # 6

    ops.append(create_const_op(g, "bias1", bias_shape, "INT32", bias_data))  # 7
    ops.append(
        create_const_op(g, "bias1_s", [1], "FLOAT32", np.array([1], dtype=np.float32))
    )  # 8enable 0
    ops.append(create_const_op(g, "bias1_z", [1], "INT32", zero))  # 9
    ops.append(
        create_dequantize_op(
            g, "dequantize-bias", "dequantize-linear", "FLOAT32", ops[7], ops[8], ops[9]
        )
    )  # 10

    ops.append(
        g.create_op(
            "conv1",
            "conv2d",
            {
                "pad_mode": "SAME",
                "kernel": [3, 3],
                "stride": [1, 1],
            },
            {
                "input": [ops[2]],
                "weights": [ops[6]],
                "bias": [ops[10]],
            },
        )
    )  # 11
    ops.append(
        create_const_op(
            g, "conv1_s", [1], "FLOAT32", np.array([100.0], dtype=np.float32)
        )
    )  # 12
    ops.append(create_const_op(g, "conv1_z", [1], "INT32", zero))  # 13
    ops.append(
        create_dequantize_op(
            g, "quantize-conv", "quantize-linear", "INT8", ops[11], ops[12], ops[13]
        )
    )  # 14
    root = g.get_root_subgraph()
    root.create_child_subgraph()
    create_user_sg(root, op_input)
    create_cpu_sg(root, ops)
    g.serialize(str(work_dir / "b.xmodel"))
    print(str(work_dir / "b.xmodel"))


random = np.random.RandomState(237)

input_shape = [1, 224, 224, 4]
input_data = random_randint(input_shape).astype(np.int8)
input_data.tofile(work_dir / "input.bin")
weight_shape = [64, 3, 3, 4]
weight_data = random_randint(weight_shape).astype(np.int32)
bias_shape = [64]
bias_data = random_randint(bias_shape).astype(np.int32)
print(f"BASE={bias_data}")
create_qconv_xmodel(input_shape, weight_shape, bias_shape, weight_data, bias_data)
create_dequantize_conv_xmodel(
    input_shape, weight_shape, bias_shape, weight_data, bias_data
)
