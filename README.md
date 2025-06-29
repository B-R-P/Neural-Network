# Neural Network from Scratch

A simple, educational implementation of a feed-forward neural network in pure Python.  
This project demonstrates core concepts—forward propagation, backpropagation, activation functions.

## Features

- Build multi-layer feed-forward networks (input, hidden, output layers)  
- Support for common activation functions:
  - Sigmoid  
  - ReLU  
  - Tanh  
  - Softmax  
- Manual forward propagation and backpropagation  
- Multi-threaded neuron computation for acceleration  

---

## Requirements

- Python 3.7+  
- NumPy  
- tqdm  
- Numba (optional, for JIT compilation speedups)  

---

## Installation

```bash
git clone https://github.com/B-R-P/Neural-Network.git
cd Neural-Network
pip install numpy tqdm numba
```

---

## Usage

Import the main module:

```python
from NN import Network, save_model, regain_model, check_err
```

### Creating a Network

```python
# Create a network with:
# - 4 inputs
# - two hidden layers with 8 and 5 neurons respectively (ReLU)
# - 3 outputs (Softmax)
net = Network(
    input_size=4,
    hidden_layers=[8, 5],
    output_size=3,
    hidden_activation='reLU',
    output_activation='softmax'
)
```

### Training

```python
# Single-sample training
x = [0.5, 0.2, 0.1, 0.7]
y_true = [0, 0, 1]  # One-hot target for class 2
net.train(x, y_true, learning_rate=0.1)

# Batch training over multiple epochs and samples
data = [([0.1,0.2,0.3,0.4], [1,0,0]), ...]
net.traina(data, iterations=1000, initial_lr=0.1)
```

### Prediction

```python
x_new = [0.6, 0.1, 0.8, 0.3]
pred_class, pred_confidence = net.predict(x_new)
print(f"Predicted class: {pred_class}, Confidence: {pred_confidence:.4f}")
```

### Saving & Loading Models

```python
# Save the trained model to ZIP
save_model(net, 'mymodel.zip')

# Later, load it back
loaded_net = regain_model('mymodel.zip')
```

---

## API Reference

### Class: `Network`

- `Network(input_size, hidden_layers, output_size, hidden_activation, output_activation)`  
  Construct a new network.

- `train(x, y, learning_rate)`  
  Train on a single input–target pair.

- `traina(data, iterations, initial_lr)`  
  Train over multiple samples and iterations. `data` is a list of `(x, y)`.

- `predict(x)` → `(class_index, confidence)`  
  Run inference and return the predicted class.

- `printer()` / `wprinter()`  
  Print network outputs or weights/biases.

- `check_err(data)` → `float`  
  Compute total error over a dataset.

### Functions

- `save_model(net, filename)`  
- `regain_model(filename)`  

### Activation functions

- `sigmoid`, `reLU`, `tanh`, `softmax` (found in `actdic`)

## Contributing

Contributions, issues, and feature requests are welcome!  
Please open a GitHub issue or submit a pull request.

## License

This project is licensed under the MIT License.  
See [LICENSE](LICENSE) for details.  
