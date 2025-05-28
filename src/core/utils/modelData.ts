import { Tensor } from "onnxruntime-web";
import { SAM_MASK_SIZE } from "../constants";

export function buildSamFeeds(params: {
  clicks: Array<{ x: number; y: number; clickType: 0 | 1 }>;
  embedding: Tensor;
  originalWidth: number;
  originalHeight: number;
}) {
  const { clicks, embedding, originalWidth, originalHeight } = params;

  // How much we scaled the img when encoding
  const samScale = 1024 / Math.max(originalWidth, originalHeight);

  const n = clicks.length;
  const pointCoords = new Float32Array(2 * (n + 1));
  const pointLabels = new Float32Array(n + 1);

  // Fill in user clicks
  for (let i = 0; i < n; i++) {
    pointCoords[2 * i] = clicks[i].x * samScale;
    pointCoords[2 * i + 1] = clicks[i].y * samScale;
    pointLabels[i] = clicks[i].clickType;
  }
  // The required “padding” point
  pointCoords[2 * n] = 0.0;
  pointCoords[2 * n + 1] = 0.0;
  pointLabels[n] = -1.0;

  const pointCoordsTensor = new Tensor("float32", pointCoords, [1, n + 1, 2]);
  const pointLabelsTensor = new Tensor("float32", pointLabels, [1, n + 1]);

  // Original image size tensor (no explicit dims)
  const origImSizeTensor = new Tensor("float32", [
    originalHeight,
    originalWidth,
  ]);

  // No prior mask
  const maskInput = new Tensor(
    "float32",
    new Float32Array(SAM_MASK_SIZE * SAM_MASK_SIZE),
    [1, 1, SAM_MASK_SIZE, SAM_MASK_SIZE]
  );
  const hasMaskInput = new Tensor("float32", [0]);

  return {
    image_embeddings: embedding,
    point_coords: pointCoordsTensor,
    point_labels: pointLabelsTensor,
    orig_im_size: origImSizeTensor,
    mask_input: maskInput,
    has_mask_input: hasMaskInput,
  };
}
