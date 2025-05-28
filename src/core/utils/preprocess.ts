import { Tensor } from "onnxruntime-web";
import { ENCODER_INPUT_SIZE } from "../constants";

/**
 * Resize & normalize an image (HTMLImageElement or Canvas) into a CHW float32 tensor,
 * padded to square [ENCODER_INPUT_SIZE x ENCODER_INPUT_SIZE].
 */
export function preprocessImageForEncoder(
  image: HTMLImageElement | HTMLCanvasElement,
  targetSize = ENCODER_INPUT_SIZE
): {
  tensor: Tensor;
  originalWidth: number;
  originalHeight: number;
} {
  let originalWidth: number;
  let originalHeight: number;
  let sourceCanvas: HTMLCanvasElement;

  if (image instanceof HTMLImageElement) {
    originalWidth = image.naturalWidth;
    originalHeight = image.naturalHeight;
    // Create a temporary canvas to draw the image for consistent processing
    sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = originalWidth;
    sourceCanvas.height = originalHeight;
    const ctx = sourceCanvas.getContext("2d")!;
    ctx.drawImage(image, 0, 0, originalWidth, originalHeight);
  } else {
    // Already a canvas, use it as the source
    sourceCanvas = image;
    originalWidth = sourceCanvas.width;
    originalHeight = sourceCanvas.height;
  }

  // Scale longest side to targetSize
  const scale = targetSize / Math.max(originalWidth, originalHeight);
  const newWidth = Math.round(originalWidth * scale);
  const newHeight = Math.round(originalHeight * scale);

  // Create a new canvas for resizing
  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = newWidth;
  resizedCanvas.height = newHeight;
  const resizedCtx = resizedCanvas.getContext("2d")!;
  resizedCtx.drawImage(sourceCanvas, 0, 0, newWidth, newHeight);

  const { data: pixelData } = resizedCtx.getImageData(
    0,
    0,
    newWidth,
    newHeight
  );

  // prepare a zero-padded CHW float32 buffer
  const channels = 3;
  const float32Data = new Float32Array(channels * targetSize * targetSize);
  float32Data.fill(0);

  // normalization constants (RGB)
  const PIXEL_MEAN = [123.675, 116.28, 103.53];
  const PIXEL_STD = [58.395, 57.12, 57.375];

  // copy + normalize into CHW
  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      const i4 = (y * newWidth + x) * 4;
      // R
      float32Data[0 * targetSize * targetSize + y * targetSize + x] =
        (pixelData[i4] - PIXEL_MEAN[0]) / PIXEL_STD[0];
      // G
      float32Data[1 * targetSize * targetSize + y * targetSize + x] =
        (pixelData[i4 + 1] - PIXEL_MEAN[1]) / PIXEL_STD[1];
      // B
      float32Data[2 * targetSize * targetSize + y * targetSize + x] =
        (pixelData[i4 + 2] - PIXEL_MEAN[2]) / PIXEL_STD[2];
    }
  }

  const tensor = new Tensor("float32", float32Data, [
    1,
    channels,
    targetSize,
    targetSize,
  ]);

  return { tensor, originalWidth, originalHeight };
}
