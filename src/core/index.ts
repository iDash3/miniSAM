import { InferenceSession, Tensor } from "onnxruntime-web";
import FULL_ENCODER_MODEL_URL from "../models/encoder.onnx";
import FULL_SAM_MODEL_URL from "../models/sam.onnx";

import { ENCODER_INPUT_SIZE } from "./constants";
import { preprocessImageForEncoder } from "./utils/preprocess";
import { buildSamFeeds } from "./utils/modelData";
import { maskToImageData } from "./utils/postprocess";

// Click types
export type ClickType = "include" | "exclude";
export interface Click {
  x: number;
  y: number;
  type: ClickType;
}

// Convert click types to SAM model format
const clickTypeToSamLabel = (type: ClickType): 0 | 1 => {
  return type === "include" ? 1 : 0;
};

// Cache for image embeddings
type ImageKey = string; // URL or other unique identifier
interface EmbeddingCache {
  embedding: Tensor;
  originalWidth: number;
  originalHeight: number;
}
const embeddingCache = new Map<ImageKey, EmbeddingCache>();

// Session state management
interface SessionState {
  imageKey: string;
  clicks: Click[];
  lastMask: ImageData | null;
}

const sessionStates = new Map<string, SessionState>();

// Track initialization status to prevent multiple simultaneous loads
let isInitializing = false;
let initializationPromise: Promise<void> | null = null;

async function loadModel(
  modelPath: string,
  options?: InferenceSession.SessionOptions
): Promise<InferenceSession> {
  try {
    const session = await InferenceSession.create(modelPath, options);
    return session;
  } catch (error) {
    console.error(
      `[tinysam] loadModel: InferenceSession.create() FAILED for ${modelPath.slice(
        0,
        40
      )}...`,
      error
    );
    throw error;
  }
}

let encoderSession: InferenceSession | null = null;
let samSession: InferenceSession | null = null;

type SessionOptions = InferenceSession.SessionOptions;

const DEFAULT_SESSION_OPTIONS: SessionOptions = {
  executionProviders: ["wasm"] as const,
  graphOptimizationLevel: "all" as const,
};

// For base64 data URLs, add the optimization flag to use bytes directly
const getSessionOptions = (
  modelPath: string,
  baseOptions: SessionOptions
): SessionOptions => {
  if (modelPath.startsWith("data:")) {
    return {
      ...baseOptions,
      extra: {
        session: {
          use_ort_model_bytes_directly: "1",
        },
      },
    };
  }
  return baseOptions;
};

export async function initSegmentation(opts?: {
  encoderModelPath?: string;
  samModelPath?: string;
  sessionOptions?: SessionOptions;
}): Promise<void> {
  // If already initializing, return the existing promise
  if (isInitializing) {
    return initializationPromise!;
  }

  // If both models are already loaded, return immediately
  if (encoderSession && samSession && !opts) {
    return;
  }

  isInitializing = true;

  // Create a promise for the initialization
  initializationPromise = (async () => {
    try {
      const options = opts || {};

      let finalEncoderModelPath =
        options.encoderModelPath || FULL_ENCODER_MODEL_URL;
      let finalSamModelPath = options.samModelPath || FULL_SAM_MODEL_URL;
      const finalSessionOptions =
        options.sessionOptions || DEFAULT_SESSION_OPTIONS;

      if (!encoderSession || options.hasOwnProperty("encoderModelPath")) {
        encoderSession = await loadModel(
          finalEncoderModelPath,
          getSessionOptions(finalEncoderModelPath, finalSessionOptions)
        );
      }

      if (!samSession || options.hasOwnProperty("samModelPath")) {
        samSession = await loadModel(
          finalSamModelPath,
          getSessionOptions(finalSamModelPath, finalSessionOptions)
        );
      }

      console.log(`[tinysam] initSegmentation: Completed successfully`);
    } catch (error) {
      console.error(`[tinysam] initSegmentation: Failed with error`, error);
      throw error;
    } finally {
      isInitializing = false;
    }
  })();

  return initializationPromise;
}

// Segmentation Session class for stateful interaction
export class SegmentationSession {
  private sessionId: string;
  private imageKey: string;

  constructor(image: HTMLImageElement | HTMLCanvasElement) {
    this.sessionId = `session_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;
    this.imageKey =
      image instanceof HTMLImageElement
        ? image.src
        : `canvas_${image.width}x${image.height}_${Date.now()}`;

    // Initialize session state
    sessionStates.set(this.sessionId, {
      imageKey: this.imageKey,
      clicks: [],
      lastMask: null,
    });
  }

  // Add a click to the current session
  addClick(x: number, y: number, type: ClickType = "include"): this {
    const state = sessionStates.get(this.sessionId);
    if (!state) {
      throw new Error("Session state not found");
    }

    const click: Click = { x, y, type };
    state.clicks.push(click);

    return this;
  }

  // Remove the last click
  removeLastClick(): this {
    const state = sessionStates.get(this.sessionId);
    if (!state) {
      throw new Error("Session state not found");
    }

    if (state.clicks.length > 0) {
      state.clicks.pop();
    }
    return this;
  }

  // Reset all clicks for this session
  reset(): this {
    const state = sessionStates.get(this.sessionId);
    if (!state) {
      throw new Error("Session state not found");
    }

    state.clicks = [];
    state.lastMask = null;

    return this;
  }

  // Get current clicks
  getClicks(): Click[] {
    const state = sessionStates.get(this.sessionId);
    if (!state) {
      throw new Error("Session state not found");
    }
    return [...state.clicks];
  }

  // Get the number of clicks
  getClickCount(): number {
    const state = sessionStates.get(this.sessionId);
    if (!state) {
      throw new Error("Session state not found");
    }
    return state.clicks.length;
  }

  // Run segmentation with current clicks
  async segment(
    image: HTMLImageElement | HTMLCanvasElement
  ): Promise<ImageData | null> {
    const state = sessionStates.get(this.sessionId);
    if (!state) {
      throw new Error("Session state not found");
    }

    if (state.clicks.length === 0) {
      state.lastMask = null;
      return null;
    }

    // Convert our clicks to the format expected by the segment function
    const samClicks = state.clicks.map((click) => ({
      x: click.x,
      y: click.y,
      clickType: clickTypeToSamLabel(click.type),
    }));

    const mask = await segment({
      image,
      clicks: samClicks,
    });

    state.lastMask = mask;
    return mask;
  }

  // Get the last generated mask
  getLastMask(): ImageData | null {
    const state = sessionStates.get(this.sessionId);
    if (!state) {
      throw new Error("Session state not found");
    }
    return state.lastMask;
  }

  // Dispose of this session and clean up resources
  dispose(): void {
    sessionStates.delete(this.sessionId);
  }
}

// Create a new segmentation session for an image
export function createSession(
  image: HTMLImageElement | HTMLCanvasElement
): SegmentationSession {
  return new SegmentationSession(image);
}

// Segment function (maintained for backward compatibility)
export async function segment(params: {
  image: HTMLImageElement | HTMLCanvasElement;
  clicks: Array<{ x: number; y: number; clickType: 0 | 1 }>;
}): Promise<ImageData> {
  if (!encoderSession || !samSession) {
    throw new Error("Please call initSegmentation() before segment().");
  }

  // Generate a key for the image (src URL for HTMLImageElement or a content hash for canvas)
  const imageKey =
    params.image instanceof HTMLImageElement
      ? params.image.src
      : `canvas_${params.image.width}x${params.image.height}_${Date.now()}`;

  // Check if we already have an embedding for this image
  let embedding: Tensor;
  let originalWidth: number;
  let originalHeight: number;

  if (embeddingCache.has(imageKey)) {
    const cached = embeddingCache.get(imageKey)!;
    embedding = cached.embedding;
    originalWidth = cached.originalWidth;
    originalHeight = cached.originalHeight;
  } else {
    // 1. Preprocess image → tensor, plus original dims
    const {
      tensor,
      originalWidth: width,
      originalHeight: height,
    } = preprocessImageForEncoder(params.image, ENCODER_INPUT_SIZE);
    originalWidth = width;
    originalHeight = height;

    // 2. Run encoder
    const startTime = performance.now();
    const encRes = await encoderSession.run({ input: tensor });
    embedding = encRes[encoderSession.outputNames[0]];
    const endTime = performance.now();

    // Store in cache
    embeddingCache.set(imageKey, {
      embedding,
      originalWidth,
      originalHeight,
    });
  }

  // 3. Prepare SAM inputs (clicks + embedding + image size)
  const samStartTime = performance.now();
  const feeds = buildSamFeeds({
    clicks: params.clicks,
    embedding,
    originalWidth,
    originalHeight,
  });

  // 4. Run SAM
  const samOut = await samSession.run(feeds);
  const maskTensor = samOut[samSession.outputNames[0]];
  const samEndTime = performance.now();

  // 5. Post-process maskTensor → ImageData
  return maskToImageData(
    maskTensor.data as Uint8Array,
    maskTensor.dims[3],
    maskTensor.dims[2]
  );
}

// Add a utility function to clear the embedding cache if needed
export function clearEmbeddingCache(): void {
  embeddingCache.clear();
}

// Clear all session states
export function clearAllSessions(): void {
  sessionStates.clear();
}

/**
 * Precompute and cache the embedding for an image.
 * Use this function to prepare an image before making it available for user interaction.
 *
 * @param image The image to process
 * @returns A promise that resolves with the image key when embedding is complete
 */
export async function precomputeEmbedding(
  image: HTMLImageElement | HTMLCanvasElement
): Promise<string> {
  if (!encoderSession) {
    throw new Error(
      "Please call initSegmentation() before precomputeEmbedding()."
    );
  }

  // Generate a key for the image
  const imageKey =
    image instanceof HTMLImageElement
      ? image.src
      : `canvas_${image.width}x${image.height}_${Date.now()}`;

  // If we already have an embedding for this image, just return the key
  if (embeddingCache.has(imageKey)) {
    return imageKey;
  }

  // Preprocess image and get dimensions
  const { tensor, originalWidth, originalHeight } = preprocessImageForEncoder(
    image,
    ENCODER_INPUT_SIZE
  );

  // Run encoder and time it
  const startTime = performance.now();
  const encRes = await encoderSession.run({ input: tensor });
  const embedding = encRes[encoderSession.outputNames[0]];
  const endTime = performance.now();

  // Store in cache
  embeddingCache.set(imageKey, {
    embedding,
    originalWidth,
    originalHeight,
  });

  return imageKey;
}
