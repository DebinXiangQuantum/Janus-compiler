import argparse
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janus.circuit import Circuit,Parameter
import numpy as np
from janus.circuit import Circuit,Parameter
import numpy as np

thetas = []
for i in range(3):
    theta = Parameter(r'$\theta_{'+str(i)+r'}$')
    thetas.append(theta)

circuit = Circuit.from_layers([
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'rz', 'qubits': [1], 'params': [2*thetas[0]]}],
    [{'name': 'cx', 'qubits': [1, 0], 'params': []}],
    [{'name': 'rz', 'qubits': [0], 'params': [2*thetas[1]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'rz', 'qubits': [1], 'params': [2*thetas[2]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'crz', 'qubits': [1, 2], 'params': [np.pi/4]}],
        [{'name': 'cx', 'qubits': [1, 0], 'params': []}],
    [{'name': 'rz', 'qubits': [0], 'params': [2*thetas[1]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'rz', 'qubits': [1], 'params': [2*thetas[2]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'crz', 'qubits': [1, 2], 'params': [np.pi/4]}],
        [{'name': 'cx', 'qubits': [1, 0], 'params': []}],
    [{'name': 'rz', 'qubits': [0], 'params': [2*thetas[1]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'rz', 'qubits': [1], 'params': [2*thetas[2]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'crz', 'qubits': [1, 2], 'params': [np.pi/4]}],    [{'name': 'cx', 'qubits': [1, 0], 'params': []}],
    [{'name': 'rz', 'qubits': [0], 'params': [2*thetas[1]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'rz', 'qubits': [1], 'params': [2*thetas[2]]}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
    [{'name': 'crz', 'qubits': [1, 2], 'params': [np.pi/4]}],
    [{'name': 'measure', 'qubits': [0,1,2], 'params': []}],
], n_qubits=3)
circuit.draw(output='png', filename='circuit.png',  dpi=600)