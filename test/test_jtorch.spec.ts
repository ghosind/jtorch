import { Tensor, isTensor, arange } from "../dist/jtorch";
import {registerHandler} from 'segfault-handler';

registerHandler('crash.log')

let tensor = new Tensor();
console.log(tensor.toString());
const t = tensor.t();
console.log(t.toString());
const tt = t.t();
console.log(tt.toString());

// @ts-ignore
console.log(isTensor(1), 1 instanceof Tensor);
console.log(isTensor(tensor), tensor instanceof Tensor);

console.log(new Tensor(tensor).toString())

tensor = arange(5);
console.log(tensor.toString());
console.log(tensor.equal(arange(5)));
