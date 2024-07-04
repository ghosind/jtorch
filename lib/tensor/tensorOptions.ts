import { type Tensor } from './index';

export interface TensorOptions {
  out?: Tensor;
  dtype?: string;
  layout?: string;
  device?: string;
  requiresGrad?: boolean;
};
