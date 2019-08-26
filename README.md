# XNORnet

[XNOR-net](https://arxiv.org/pdf/1603.05279.pdf) implementation using PyTorch.

```

|Dataset| Network | 1bit | 32bit | Acc of XNOR | Acc of 32bit|
|-------|---------|------|-------|-------------|-------------|
|MNIST  | LeNet5  | 0.425| 0.007 | 99.30       | 99.34       |
|CIFAR10| NIN     | 0.949| 0.025 | 86.03       | 89.67       |
```



### Implementation



Pack XNORConv2d and XNORLinear as two modules, you can add them any where you want. Remember to call the following function after ```loss.backward()```



```python
        for layer in model.children():
            if isinstance(layer, XNORConv2d) or isinstance(layer, XNORLinear):
                layer.copy_grad()
```

### MNIST

Main function modified from [pytorch examples](https://github.com/pytorch/examples/tree/master/mnist). BNN is sensitive to hyper-parameters, so be patience!

Network structure based on LeNet5.

### CIFAR10

Network structure based on [NIN](https://arxiv.org/pdf/1312.4400.pdf)

### Acknowlegement

Thanks to jiecaoyu's implementation [XNOR-Net-PyTorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch)
