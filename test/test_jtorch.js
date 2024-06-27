const { Tensor } = require("../dist/jtorch");
const SegHandler = require('segfault-handler');
SegHandler.registerHandler('crash.log')

const tensor = new Tensor();
console.log(tensor.toString());
const t = tensor.t();
console.log(t.toString());
const tt = t.t();
console.log(tt.toString());
