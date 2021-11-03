import numpy as np

def apply_operation(a, b, op):
    if op == '+':
        result = a + b
    elif op == '-':
        result = a - b
    elif op == '*':
        result = a * b
    elif op == '/':
        if b == 0: 
            return np.Inf
        result = a / b
    return result

def map_label():
    values = []
    inputs = []
    for num1 in range(10):
        for num2 in range(10):
            for op in ['+', '-', '*', '/']:
                result = apply_operation(num1, num2, op)
                values.append(result)
                inputs.append((num1, num2, op))

    label_to_value = dict(zip(range(96),list(np.unique(values))))
    return label_to_value

