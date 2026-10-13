import assert from 'node:assert/strict';
import { commitInset } from '../../render/render3D/src/inset.ts';

let width = 960, height = 640, resets = 0;
const canvas = {
  get width() { return width; }, set width(value) { width = value; resets++; },
  get height() { return height; }, set height(value) { height = value; resets++; },
  style: {}, dataset: {}, hidden: true,
};
const drawn = [];
const context = { drawImage(image, x, y) { drawn.push([image.naturalWidth, image.naturalHeight, x, y]); } };
commitInset(canvas, context, { naturalWidth: 960, naturalHeight: 640 }, 0);
assert.equal(resets, 0);
commitInset(canvas, context, { naturalWidth: 640, naturalHeight: 512 }, 1);
assert.equal(resets, 2);
assert.equal(canvas.style.aspectRatio, '640 / 512');
commitInset(canvas, context, { naturalWidth: 640, naturalHeight: 512 }, 2);
assert.equal(resets, 2); // Consecutive frames never clear/resize the display.
assert.deepEqual(drawn.at(-1), [640, 512, 0, 0]);
assert.equal(canvas.dataset.tick, '2');
assert.equal(canvas.hidden, false);
commitInset(canvas, context, { naturalWidth: 960, naturalHeight: 640 }, 3);
assert.equal(resets, 4);
assert.equal(canvas.style.aspectRatio, '960 / 640');
console.log('PASS: inset map dimensions, aspect ratio and stable frame commits');
