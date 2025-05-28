/**
 * Turn a flat mask (Float32Array or Uint8Array of length w*h) into ImageData.
 */
export function maskToImageData(
  mask: Float32Array | Uint8Array,
  maskWidth: number,
  maskHeight: number
): ImageData {
  const rgba = new Uint8ClampedArray(4 * maskWidth * maskHeight);
  // black mask with alpha=255 where mask>0
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0) {
      rgba[4 * i + 3] = 255;
    }
  }
  return new ImageData(rgba, maskWidth, maskHeight);
}
