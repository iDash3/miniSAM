# miniSAM: Plug and play ONNX SAM segmentation in the browser

miniSAM is a JavaScript library for performing image segmentation directly in the browser using ONNX models. It uses a lightweight, distilled version of Meta AI's original Segment Anything Model (SAM), specifically optimized for efficient in-browser execution.

The training code used for the model distillation process will be made available soon. miniSAM provides a stateful API for interactive segmentation by allowing users to add include/exclude clicks on an image.

## Features

- **In-Browser Segmentation:** Runs entirely on the client-side, no server needed.
- **ONNX Runtime:** `onnxruntime-web` for efficient model inference.
- **Stateful Segmentation Sessions:** Manage clicks and masks per image using a `SegmentationSession`.
- **Click-Based Interaction:** Supports positive (include) and negative (exclude) clicks.
- **Embedding Cache:** Caches image embeddings to speed up subsequent operations on the same image.
- **Customizable Model Paths:** Allows specifying paths to your SAM encoder and decoder ONNX models.

## Installation

Install miniSAM and its peer dependency `onnxruntime-web` using npm or yarn:

```bash
npm install minisam onnxruntime-web
# or
yarn add minisam onnxruntime-web
```

## Quick Start

Here's a basic example of how to use miniSAM:

```javascript
import {
  initSegmentation,
  createSession,
  precomputeEmbedding,
  ClickType,
} from "minisam";

async function runSegmentation(myImageElement, clickPoints) {
  try {
    // Initialize miniSAM (loads models)
    await initSegmentation({
      // To use default CDN models (recommended):
      // encoderModelPath: DEFAULT_ENCODER_MODEL_PATH,
      // samModelPath: DEFAULT_SAM_MODEL_PATH,
      // For custom local/CDN models:
      // encoderModelPath: '/path/to/your/encoder.onnx',
      // samModelPath: '/path/to/your/sam_decoder.onnx'
    });
    console.log("miniSAM initialized!");

    // Precompute image embedding for faster interaction
    await precomputeEmbedding(myImageElement);
    console.log("Embedding precomputed for the image.");

    // Create a segmentation session for the image
    const session = createSession(myImageElement);
    console.log("Segmentation session created.");

    // Add clicks
    clickPoints.forEach((p) => {
      session.addClick(p.x, p.y, p.type);
    });

    console.log(`Added ${session.getClickCount()} clicks.`);

    // Perform segmentation
    const imageDataMask = await session.segment(myImageElement);

    if (imageDataMask) {
      console.log("Segmentation successful! Mask generated:", imageDataMask);

      // Example: Draw mask on canvas
      const ctx = myDisplayCanvas.getContext('2d');
      myDisplayCanvas.width = imageDataMask.width;
      myDisplayCanvas.height = imageDataMask.height;
      ctx.putImageData(imageDataMask, 0, 0);

      // Example: Overlay on existing canvas
      const tempMaskCanvas = document.createElement('canvas');
      tempMaskCanvas.width = imageDataMask.width;
      tempMaskCanvas.height = imageDataMask.height;
      tempMaskCanvas.getContext('2d').putImageData(imageDataMask, 0, 0);
      mainDisplayCtx.globalAlpha = 0.5;
      mainDisplayCtx.drawImage(tempMaskCanvas, 0, 0, originalImageWidth, originalImageHeight);
    } else {
      console.log("No mask generated.");
    }

    // Optional: Session cleanup
    session.dispose();
  } catch (error) {
    console.error("Error during segmentation:", error);
  }
}

// Example usage:
const imageEl = document.getElementById('my-image');
const exampleClicks = [
  { x: 100, y: 150, type: "include" as ClickType },
  { x: 200, y: 250, type: "exclude" as ClickType }
];
runSegmentation(imageEl, exampleClicks);
```

## API Reference

### `initSegmentation(opts?: InitializationOptions): Promise<void>`

Initializes the segmentation engine by loading the ONNX models. This must be called before any other miniSAM functions. See the "Model Loading" section for details on how models are loaded and how to customize paths.

- `opts` (optional): `InitializationOptions` object.
  - `encoderModelPath?: string`: URL or path to the encoder ONNX model. If not provided, defaults to the CDN path `DEFAULT_ENCODER_MODEL_PATH`.
  - `samModelPath?: string`: URL or path to the SAM (decoder) ONNX model. If not provided, defaults to the CDN path `DEFAULT_SAM_MODEL_PATH`.
  - `sessionOptions?: InferenceSession.SessionOptions`: Advanced ONNX Runtime session options.

### `precomputeEmbedding(image: HTMLImageElement | HTMLCanvasElement): Promise<string>`

Precomputes and caches the embedding for a given image. This can significantly speed up the first call to `session.segment()` or the legacy `segment()` function for that image.

- `image`: The `HTMLImageElement` or `HTMLCanvasElement` to process.
- Returns: A `Promise` that resolves with a unique `imageKey` (string) for the processed image, which is used internally for caching.

### `createSession(image: HTMLImageElement | HTMLCanvasElement): SegmentationSession`

Creates a new stateful `SegmentationSession` for a specific image.

- `image`: The `HTMLImageElement` or `HTMLCanvasElement` this session will be associated with. The embedding for this image will be computed on the first `segment()` call if not already cached or precomputed.

### `segment(params: SegmentParams): Promise<ImageData>` (Legacy)

A stateless function to perform segmentation. It's recommended to use `SegmentationSession` for new implementations.
This function will compute or retrieve a cached embedding for the image, then run segmentation.

- `params`: `SegmentParams` object.
  - `image: HTMLImageElement | HTMLCanvasElement`: The image to segment.
  - `clicks: Array<{ x: number, y: number, clickType: 0 | 1 }>`: Array of click objects. `clickType: 1` for include, `0` for exclude.
- Returns: A `Promise` that resolves with an `ImageData` object representing the mask.

### `clearEmbeddingCache(): void`

Clears all cached image embeddings.

### `clearAllSessions(): void`

Clears all active `SegmentationSession` states. Note: This does not call `dispose()` on individual sessions but rather clears the central store.

---

## Mask Format and Extraction

**Important:** miniSAM returns masks in RGBA ImageData format where the mask information is stored in the alpha channel, not the RGB channels. When processing the returned ImageData object from `session.segment()`:

- **Alpha channel (A) = 255:** Foreground pixel (included in the mask)
- **Alpha channel (A) = 0:** Background pixel (excluded from the mask)
- **RGB channels:** Typically all 0 (black) and should be ignored for mask logic

This differs from traditional grayscale masks where the mask information is stored in the RGB channels. When extracting objects or applying masks to images, always check the alpha channel values rather than RGB values. For example:

```javascript
const maskData = imageDataMask.data;
for (let i = 0; i < maskData.length; i += 4) {
  const alpha = maskData[i + 3];
  if (alpha === 255) {
    console.log("Foreground pixel at:", Math.floor(i / 4));
  }
}
```

## Model Behavior

- The library expects SAM-compatible ONNX models: an image encoder and a mask decoder.
- Image preprocessing scales the longest side of the input image to 1024px and pads it to a square tensor for the encoder.
- Click coordinates are automatically scaled to match the preprocessed image dimensions.
- The output mask is an `ImageData` object, typically 256x256, which can then be upscaled and drawn onto a canvas.

## Development & Building

If you are working on the `miniSAM` library itself:

- Install dependencies: `npm install`
- Build the library: `npm run build`
  This command cleans the `dist` folder, runs webpack to bundle the library, and then uses `npm pack` to create a `.tgz` tarball in the `miniSAM` root directory (e.g., `minisam-0.2.0.tgz`). This tarball can be installed locally by other projects.
