const { Tensor, isTensor } = require("../dist/jtorch");
const SegHandler = require('segfault-handler');
SegHandler.registerHandler('crash.log')

const tensor = new Tensor();
console.log(tensor.toString());
const t = tensor.t();
console.log(t.toString());
const tt = t.t();
console.log(tt.toString());

console.log(isTensor(1), 1 instanceof Tensor);
console.log(isTensor(tensor), tensor instanceof Tensor);

console.log(new Tensor(tensor).toString())
