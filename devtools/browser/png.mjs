// Minimal PNG reader for browser screenshots:
// 8-bit non-interlaced RGB or RGBA as Chrome emits, anything else is rejected.
// A coherent blob of differing pixels is a regression, scattered anti-aliasing noise is not.

import { inflateSync } from "node:zlib";

const kSignature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const paethPredictor = (left, above, upperLeft) => {
  const estimate = left + above - upperLeft;
  const distanceLeft = Math.abs(estimate - left);
  const distanceAbove = Math.abs(estimate - above);
  const distanceUpperLeft = Math.abs(estimate - upperLeft);
  if (distanceLeft <= distanceAbove && distanceLeft <= distanceUpperLeft) {
    return left;
  }
  return distanceAbove <= distanceUpperLeft ? above : upperLeft;
};

/// decodes a PNG into { width, height, rgba }, rgba being 4 bytes per pixel in row order
export const decodePng = (buffer) => {
  if (!buffer.subarray(0, 8).equals(kSignature)) {
    throw new Error("not a PNG");
  }
  let width = 0;
  let height = 0;
  let channels = 0;
  const parts = [];

  for (let offset = 8; offset < buffer.length;) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    offset += 12 + length; // length, type, data, CRC

    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      const depth = data.readUInt8(8);
      const colourType = data.readUInt8(9);
      const interlace = data.readUInt8(12);
      if (depth !== 8 || interlace !== 0 || (colourType !== 2 && colourType !== 6)) {
        throw new Error(`unsupported PNG: depth ${depth}, colour type ${colourType}, interlace ${interlace}`);
      }
      channels = colourType === 6 ? 4 : 3;
    } else if (type === "IDAT") {
      parts.push(data);
    } else if (type === "IEND") {
      break;
    }
  }

  const raw = inflateSync(Buffer.concat(parts));
  const stride = width * channels;
  const rgba = Buffer.alloc(width * height * 4);
  const previous = Buffer.alloc(stride);
  const current = Buffer.alloc(stride);

  for (let row = 0; row < height; ++row) {
    const filter = raw[row * (stride + 1)];
    raw.copy(current, 0, row * (stride + 1) + 1, row * (stride + 1) + 1 + stride);
    for (let index = 0; index < stride; ++index) {
      const left = index >= channels ? current[index - channels] : 0;
      const above = previous[index];
      const upperLeft = index >= channels ? previous[index - channels] : 0;
      const correction =
        filter === 1 ? left : filter === 2 ? above : filter === 3 ? (left + above) >> 1 : filter === 4 ? paethPredictor(left, above, upperLeft) : 0;
      current[index] = (current[index] + correction) & 0xff;
    }
    for (let column = 0; column < width; ++column) {
      const source = column * channels;
      const destination = (row * width + column) * 4;
      rgba[destination] = current[source];
      rgba[destination + 1] = current[source + 1];
      rgba[destination + 2] = current[source + 2];
      rgba[destination + 3] = channels === 4 ? current[source + 3] : 0xff;
    }
    current.copy(previous);
  }
  return { width, height, rgba };
};

export const pixelAt = (image, x, y) => {
  const offset = (y * image.width + x) * 4;
  return { r: image.rgba[offset], g: image.rgba[offset + 1], b: image.rgba[offset + 2], a: image.rgba[offset + 3] };
};

export const describePixel = (pixel) => `rgba(${pixel.r}, ${pixel.g}, ${pixel.b}, ${pixel.a})`;

export const pixelMatches = (pixel, expected, channelDelta = 12) =>
  Math.abs(pixel.r - expected.r) <= channelDelta && Math.abs(pixel.g - expected.g) <= channelDelta && Math.abs(pixel.b - expected.b) <= channelDelta;

/// largest connected blob of differing pixels, the primary regression criterion
export const compareImages = (left, right, { channelDelta = 12, clusterArea = 24 } = {}) => {
  if (left.width !== right.width || left.height !== right.height) {
    return { comparable: false, reason: `size ${left.width}x${left.height} against ${right.width}x${right.height}` };
  }
  const { width, height } = left;
  const differing = new Uint8Array(width * height);
  let differingCount = 0;
  for (let index = 0; index < width * height; ++index) {
    const offset = index * 4;
    const delta = Math.max(
      Math.abs(left.rgba[offset] - right.rgba[offset]),
      Math.abs(left.rgba[offset + 1] - right.rgba[offset + 1]),
      Math.abs(left.rgba[offset + 2] - right.rgba[offset + 2]),
    );
    if (delta > channelDelta) {
      differing[index] = 1;
      ++differingCount;
    }
  }

  let largestBlob = 0;
  const stack = [];
  for (let seed = 0; seed < differing.length; ++seed) {
    if (differing[seed] !== 1) {
      continue;
    }
    let area = 0;
    stack.push(seed);
    differing[seed] = 2;
    while (stack.length > 0) {
      const index = stack.pop();
      ++area;
      const x = index % width;
      const y = (index - x) / width;
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
          continue;
        }
        const neighbour = ny * width + nx;
        if (differing[neighbour] === 1) {
          differing[neighbour] = 2;
          stack.push(neighbour);
        }
      }
    }
    largestBlob = Math.max(largestBlob, area);
  }

  return { comparable: true, differingCount, differingFraction: differingCount / (width * height), largestBlob, regression: largestBlob >= clusterArea };
};
