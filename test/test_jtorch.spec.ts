/// <reference types="mocha" />
import { expect } from 'chai';
import { Tensor, isTensor, arange, isFloatingPoint, isNonzero } from "../dist/jtorch";

describe('jtorch core', () => {
	it('constructors and type helpers', () => {
		const empty = new Tensor();
		expect(isTensor(1)).to.be.false;
		expect(isTensor(empty)).to.be.true;

		const copy = new Tensor(empty);
		expect(isTensor(copy)).to.be.true;
		expect(copy.toString()).to.equal(empty.toString());
	});

	it('arange, size, transpose and equal', () => {
		const a1 = arange(5);
		const a2 = arange(0, 5);
		expect(isTensor(a1)).to.be.true;
		expect(a1.equal(a2)).to.be.true;
		expect(a1.size(0)).to.equal(5);

		const t = a1.t();
		const tt = t.t();
		expect(tt.toString()).to.equal(a1.toString());
	});

	it('helper return types and single-element isNonzero', () => {
		const floatFlag = isFloatingPoint(arange(5));
		expect(typeof floatFlag).to.equal('boolean');

		const single = arange(1);
		const nonzeroFlag = isNonzero(single);
		expect(typeof nonzeroFlag).to.equal('boolean');
	});

	it('empty tensor copy equality', () => {
		const empty = new Tensor();
		const emptyCopy = new Tensor(empty);
		expect(emptyCopy.toString()).to.equal(empty.toString());
	});
});
