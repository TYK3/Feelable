// Client-side downscale before upload (CONTRACT-v2 §4, audit F12/F13).
// Phone photos are 12–48 MP; we send at most 2000 px on the long side as JPEG q0.9 (~0.5–1.5 MB),
// with EXIF orientation applied by the browser.

export const MAX_SIDE = 2000;
export const MAX_BYTES = 15 * 1024 * 1024;

export class IntakeError extends Error {
  constructor(public code: "heic" | "not_image" | "empty" | "too_big" | "decode", message: string) {
    super(message);
  }
}

const OK_EXT = /\.(png|jpe?g|webp|gif|bmp|tiff?|heic|heif|avif)$/i;
export const isHeic = (f: File) => /\.(heic|heif)$/i.test(f.name) || /heic|heif/i.test(f.type);

export const HEIC_MSG =
  "This iPhone photo is in HEIC format, which this browser can't open. Take a screenshot of the photo and upload that, or set Camera › Formats to “Most Compatible”.";

/** Validate cheap things first, then decode + downscale. Returns a JPEG blob (or the original file when it is
 *  already small and the browser can't re-encode it). */
export async function prepareUpload(file: File): Promise<{ blob: Blob; name: string; width: number; height: number; resized: boolean }> {
  if (file.size === 0) throw new IntakeError("empty", "That file is empty. Try choosing it again.");
  const looksImage = file.type ? file.type.startsWith("image/") : OK_EXT.test(file.name);
  if (!looksImage && !isHeic(file)) {
    throw new IntakeError("not_image", "That file isn't an image. Choose a photo or screenshot of a diagram (PNG, JPEG or WebP).");
  }
  if (file.size > 60 * 1024 * 1024) {
    throw new IntakeError("too_big", `That file is ${(file.size / 1024 / 1024).toFixed(0)} MB, too big to open here. Try a screenshot instead.`);
  }

  let bmp: ImageBitmap | null = null;
  try {
    bmp = await createImageBitmap(file, { imageOrientation: "from-image" } as ImageBitmapOptions);
  } catch {
    bmp = null;
  }
  if (!bmp) {
    // HEIC on browsers without a decoder: the server can decode HEIC (CONTRACT-v2 §1), so send the original
    // when it fits; the server reports a clear error otherwise.
    if (isHeic(file) && file.size <= MAX_BYTES) return { blob: file, name: file.name, width: 0, height: 0, resized: false };
    if (isHeic(file)) throw new IntakeError("heic", HEIC_MSG);
    throw new IntakeError("decode", "We couldn't open that image. It may be damaged — try another photo or a screenshot.");
  }

  const { width, height } = bmp;
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const smallEnough = scale === 1 && file.size <= 3 * 1024 * 1024 && /png|jpe?g|webp/i.test(file.type);
  if (smallEnough) {
    bmp.close?.();
    return { blob: file, name: file.name || "photo.png", width, height, resized: false };
  }
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new IntakeError("decode", "This browser couldn't prepare the photo. Try a screenshot instead.");
  ctx.fillStyle = "#fff"; // transparent PNGs become white, not black, in JPEG
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  const blob: Blob | null = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
  if (!blob) throw new IntakeError("decode", "This browser couldn't prepare the photo. Try a screenshot instead.");
  const base = (file.name || "photo").replace(/\.[^.]+$/, "");
  return { blob, name: `${base}.jpg`, width: w, height: h, resized: true };
}
