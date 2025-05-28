# TinySAM: Plug and play ONNX SAM segmentation in the browser

TinySAM is a JavaScript library for performing image segmentation directly in the browser using ONNX models. It is a lightweight, distilled version of Meta AI's original Segment Anything Model (SAM), specifically optimized for efficient in-browser execution.

The training code used for the model distillation process will be made available soon. TinySAM provides a stateful API for interactive segmentation by allowing users to add include/exclude clicks on an image.

## Features

- **In-Browser Segmentation:** Runs entirely on the client-side, no server needed.
- **ONNX Runtime:** `onnxruntime-web` for efficient model inference.
- **Stateful Segmentation Sessions:** Manage clicks and masks per image using a `SegmentationSession`.
- **Click-Based Interaction:** Supports positive (include) and negative (exclude) clicks.
- **Embedding Cache:** Caches image embeddings to speed up subsequent operations on the same image.
- **Customizable Model Paths:** Allows specifying paths to your SAM encoder and decoder ONNX models.

## Installation

Install TinySAM and its peer dependency `onnxruntime-web` using npm or yarn:

```bash
npm install tinysam onnxruntime-web
# or
yarn add tinysam onnxruntime-web
```

## Quick Start

Here's a basic example of how to use TinySAM:

```javascript
import {
  initSegmentation,
  createSession,
  precomputeEmbedding,
  ClickType,
} from "tinysam";

// Assuming you have an HTMLImageElement or HTMLCanvasElement `myImageElement`
// and a canvas `myDisplayCanvas` to draw on.

async function runSegmentation(myImageElement, clickPoints) {
  try {
    // 1. Initialize TinySAM (loads models)
    // By default, models are now loaded from CDN.
    // You can still override with local paths if needed.
    // See "Model Loading" section for more details.
    await initSegmentation({
      // To use default CDN models (recommended):
      // No specific paths needed, or use:
      // encoderModelPath: DEFAULT_ENCODER_MODEL_PATH,
      // samModelPath: DEFAULT_SAM_MODEL_PATH,
      // For custom local/CDN models:
      // encoderModelPath: '/path/to/your/encoder.onnx',
      // samModelPath: '/path/to/your/sam_decoder.onnx'
    });
    console.log("TinySAM initialized!");

    // 2. (Optional but Recommended) Precompute image embedding for faster interaction
    // This is useful if you load an image and want to prepare it before user clicks.
    await precomputeEmbedding(myImageElement);
    console.log("Embedding precomputed for the image.");

    // 3. Create a segmentation session for the image
    const session = createSession(myImageElement);
    console.log("Segmentation session created.");

    // 4. Add clicks
    // Clicks are { x: number, y: number, type: ClickType }
    // x, y are coordinates relative to the original image dimensions.
    // type can be "include" or "exclude".
    clickPoints.forEach((p) => {
      session.addClick(p.x, p.y, p.type);
    });

    console.log(`Added ${session.getClickCount()} clicks.`);

    // 5. Perform segmentation
    // Pass the image element again (can be the same one used for session creation/precomputation)
    const imageDataMask = await session.segment(myImageElement);

    if (imageDataMask) {
      console.log("Segmentation successful! Mask generated:", imageDataMask);
      // imageDataMask is an ImageData object. You can draw it on a canvas:
      // const ctx = myDisplayCanvas.getContext('2d');
      // myDisplayCanvas.width = imageDataMask.width;
      // myDisplayCanvas.height = imageDataMask.height;
      // ctx.putImageData(imageDataMask, 0, 0);

      // Or, for overlaying (as done in the test app):
      // const tempMaskCanvas = document.createElement('canvas');
      // tempMaskCanvas.width = imageDataMask.width;
      // tempMaskCanvas.height = imageDataMask.height;
      // tempMaskCanvas.getContext('2d').putImageData(imageDataMask, 0, 0);
      // mainDisplayCtx.globalAlpha = 0.5; // Example transparency
      // mainDisplayCtx.drawImage(tempMaskCanvas, 0, 0, originalImageWidth, originalImageHeight);
    } else {
      console.log("No mask generated (e.g., no clicks).");
    }

    // 6. Session cleanup (optional, if you are done with this specific image session)
    // session.dispose();
  } catch (error) {
    console.error("Error during segmentation:", error);
  }
}

// Example usage:
// const imageEl = document.getElementById('my-image');
// const exampleClicks = [
//   { x: 100, y: 150, type: "include" as ClickType },
//   { x: 200, y: 250, type: "exclude" as ClickType }
// ];
// runSegmentation(imageEl, exampleClicks);
```

## API Reference

### `initSegmentation(opts?: InitializationOptions): Promise<void>`

Initializes the segmentation engine by loading the ONNX models. This must be called before any other TinySAM functions. See the "Model Loading" section for details on how models are loaded and how to customize paths.

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

## Model Loading: CDN by Default, Custom Paths Supported

TinySAM is designed for ease of use and performance. By default, it loads the necessary ONNX models (`encoder.onnx` and `sam.onnx`) from a CDN. This approach keeps your application bundle small and uses browser caching for faster load times for your users.

When you initialize TinySAM without providing specific model paths, it will use default CDN URLs:

```typescript
import {
  initSegmentation,
  DEFAULT_ENCODER_MODEL_PATH, // Default CDN path for the encoder
  DEFAULT_SAM_MODEL_PATH, // Default CDN path for the SAM decoder
} from "tinysam";

async function initialize() {
  // Initializes with default models from CDN
  await initSegmentation();
  // OR, explicitly using the default paths (equivalent to the above):
  // await initSegmentation({
  //   encoderModelPath: DEFAULT_ENCODER_MODEL_PATH,
  //   samModelPath:     DEFAULT_SAM_MODEL_PATH,
  // });
  console.log("TinySAM initialized with models from CDN!");
}

initialize();
```

**Using Your Own Models (Local or Custom CDN):**

If you need to use specific versions of the ONNX models, or if you prefer to host them yourself (either locally or on your own CDN), you can easily override the default behavior by providing the paths in the `initSegmentation` options:

```typescript
import { initSegmentation } from "tinysam";

async function initializeWithCustomModels() {
  await initSegmentation({
    encoderModelPath: "/path/to/your/local/or/custom_cdn/encoder.onnx",
    samModelPath: "/path/to/your/local/or/custom_cdn/sam_decoder.onnx",
  });
  console.log("TinySAM initialized with custom models!");
}

initializeWithCustomModels();
```

This flexibility allows you to manage model deployment according to your project's specific needs while still benefiting from TinySAM's core segmentation capabilities.

---

### `SegmentationSession` Class

An instance of this class is returned by `createSession()`.

#### `constructor(image: HTMLImageElement | HTMLCanvasElement)`

(Internal: Use `createSession()` to get an instance)

#### `addClick(x: number, y: number, type: ClickType = "include"): this`

Adds a click to the session.

- `x`, `y`: Coordinates on the original image.
- `type`: `"include"` or `"exclude"`. Defaults to `"include"`.
- Returns: The session instance for chaining.

#### `removeLastClick(): this`

Removes the most recently added click.

- Returns: The session instance for chaining.

#### `reset(): this`

Clears all clicks and the last generated mask for this session.

- Returns: The session instance for chaining.

#### `getClicks(): Click[]`

Returns an array of all current clicks in the session.

- `Click`: `{ x: number, y: number, type: ClickType }`

#### `getClickCount(): number`

Returns the number of current clicks.

#### `async segment(image: HTMLImageElement | HTMLCanvasElement): Promise<ImageData | null>`

Runs segmentation using the current set of clicks for the session.

- `image`: The image element (should typically be the same one the session was created with or one with identical content and dimensions).
- Returns: A `Promise` that resolves with an `ImageData` object for the mask, or `null` if there are no clicks.

#### `getLastMask(): ImageData | null`

Returns the last `ImageData` mask generated by `session.segment()`, or `null` if no segmentation has been run or it was reset.

#### `dispose(): void`

Releases the resources and state associated with this specific session from the central session store. Call this when you are finished with a session to free up memory.

---

### Types

#### `ClickType: "include" | "exclude"`

Type alias for click types.

#### `Click: { x: number, y: number, type: ClickType }`

Interface for a click object.

## Model Behavior

- The library expects SAM-compatible ONNX models: an image encoder and a mask decoder.
- Image preprocessing scales the longest side of the input image to 1024px (configurable by `ENCODER_INPUT_SIZE` if you modify constants, but not recommended without changing model expectations) and pads it to a square tensor for the encoder.
- Click coordinates are automatically scaled to match the preprocessed image dimensions.
- The output mask is an `ImageData` object, typically 256x256 (or `SAM_MASK_SIZE` x `SAM_MASK_SIZE`), which can then be upscaled and drawn onto a canvas.

## Development & Building

If you are working on the `tinysam` library itself:

- Install dependencies: `npm install`
- Build the library: `npm run build`
  This command cleans the `dist` folder, runs webpack to bundle the library, and then uses `npm pack` to create a `.tgz` tarball in the `tinysam` root directory (e.g., `tinysam-0.1.0.tgz`). This tarball can be installed locally by other projects.

This doc has been written with AI analysis in mind.
