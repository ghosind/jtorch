const jtorch = require('bindings')('jtorch');

interface Tensor {
  t(): Tensor;
  toString(): string;
}

export const Tensor: {
  new(): Tensor,
} = jtorch.Tensor;
