/** Resize only when switching map dimensions; commit the complete image in one task. */
export function commitInset(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D,
                            image: HTMLImageElement, tick: number) {
  const width = image.naturalWidth, height = image.naturalHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    canvas.style.aspectRatio = `${width} / ${height}`;
  }
  context.drawImage(image, 0, 0);
  canvas.dataset.tick = String(tick);
  canvas.hidden = false;
}
