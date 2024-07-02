const jtorch = require('bindings')('jtorch');

interface Tensor {
  get T(): Tensor;

  abs(): Tensor;
  absolute(): Tensor;
  acos(): Tensor;
  arccos(): Tensor;
  adjoint(): Tensor;
  angle(): Tensor;
  asin(): Tensor;
  arcsin(): Tensor;
  atan(): Tensor;
  arctan(): Tensor;
  ceil(): Tensor;
  clone(): Tensor;
  conj(): Tensor;
  conjPhysical(): Tensor;
  resolveConj(): Tensor;
  resolveNeg(): Tensor;
  cos(): Tensor;
  corrcoef(): Tensor;
  cosh(): Tensor;
  acosh(): Tensor;
  arccosh(): Tensor;
  det(): Tensor;
  digamma(): Tensor;
  equal(tensor: Tensor): boolean;
  erf(): Tensor;
  erfc(): Tensor;
  erfinv(): Tensor;
  exp(): Tensor;
  expm1(): Tensor;
  greater(other: Tensor | number): Tensor;
  greaterEqual(other: Tensor | number): Tensor;
  fix(): Tensor;
  fliplr(): Tensor;
  flipud(): Tensor;
  floor(): Tensor;
  frac(): Tensor;
  i0(): Tensor;
  indices(): Tensor;
  intRepr(): Tensor;
  inverse(): Tensor;
  isfinite(): Tensor;
  isinf(): Tensor;
  isposinf(): Tensor;
  isneginf(): Tensor;
  isnan(): Tensor;
  isComplex(): boolean;
  isConj(): boolean;
  isFloatingPoint(): boolean;
  isNonzero(): boolean;
  less(other: Tensor | number): Tensor;
  lessEqual(other: Tensor | number): Tensor;
  lgamma(): Tensor;
  log(): Tensor;
  logdet(): Tensor;
  log10(): Tensor;
  log1p(): Tensor;
  log2(): Tensor;
  logit(): Tensor;
  neg(): Tensor;
  negative(): Tensor;
  pinMemory(): Tensor;
  pinverse(): Tensor;
  positive(): Tensor;
  rad2deg(): Tensor;
  ravel(): Tensor;
  reciprocal(): Tensor;
  rsqrt(): Tensor;
  sigmoid(): Tensor;
  sign(): Tensor;
  signbit(): Tensor;
  sgn(): Tensor;
  sin(): Tensor;
  sinc(): Tensor;
  sinh(): Tensor;
  asinh(): Tensor;
  arcsinh(): Tensor;
  size(dim: number): number;
  sqrt(): Tensor;
  square(): Tensor;
  t(): Tensor;
  tan(): Tensor;
  tanh(): Tensor;
  arctanh(): Tensor;
  toString(): string;
  trace(): Tensor;
  trunc(): Tensor;
  values(): Tensor;
}

export const Tensor: {
  new(data?: Tensor): Tensor;
} = jtorch.Tensor;

export const {
  isComplex,
  isConj,
  isFloatingPoint,
  isNonzero,
  isTensor,
}: {
  isComplex(v: Tensor): boolean;
  isConj(v: Tensor): boolean;
  isFloatingPoint(v: Tensor): boolean;
  isNonzero(v: Tensor): boolean;
  isTensor(v: any): boolean;
} = jtorch;
