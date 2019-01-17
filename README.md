# XNORnet

[XNOR-net](https://arxiv.org/pdf/1603.05279.pdf) implementation using PyTorch.

```

|Dataset| Network | 1bit | 32bit | Accuracy of XNOR | Acc of 32bit|
|-------|---------|------|-------|------------------|-------------|
|MNIST  | LeNet5  | 0.425| 0.007 | 99.21            | 99.34       |
```

### Implementation

Pack XNORConv2d and XNORLinear as two modules, you can add them any where you want. Remember to call the following function after ```loss.backward()```

```python
        for layer in model.children():
            if isinstance(layer, XNORConv2d) or isinstance(layer, XNORLinear):
                layer.copy_grad()
```

### MNIST

MNIST main function modified from [pytorch examples](https://github.com/pytorch/examples/tree/master/mnist). BNN is sensitive to hyper-parameters, so be patience!

### Acknowlegement

Thanks to jiecaoyu's implementation [XNOR-Net-PyTorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch)
