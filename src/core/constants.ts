export const VERSION = "0.2.0";

const CDN_BASE = `https://cdn.jsdelivr.net/npm/tinysam@${VERSION}/dist`;
// const CDN_BASE = `https://unpkg.com/tinysam@${VERSION}/dist`;

export const DEFAULT_ENCODER_MODEL_PATH = `${CDN_BASE}/encoder.onnx`;
export const DEFAULT_SAM_MODEL_PATH = `${CDN_BASE}/sam.onnx`;

// sizes for preprocess & modelData
export const ENCODER_INPUT_SIZE = 1024;
export const SAM_MASK_SIZE = 256;
