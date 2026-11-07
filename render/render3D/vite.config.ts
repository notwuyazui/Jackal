import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url));
export default defineConfig({
  root: path.join(root, 'src'),
  publicDir: path.join(root, 'public'),
  cacheDir: path.join(root, 'node_modules/.vite-jackal'),
  resolve: { alias: { '@cot': path.join(root, 'vendor/cot/src') } },
  // All runtime dependencies are ESM. Explicitly prepare the lazy 3D imports
  // so first contact with a fleet/map module cannot trigger a full-page reload.
  optimizeDeps: {
    noDiscovery: true,
    include: [
      'three',
      'three/addons/geometries/ConvexGeometry.js',
      'three/addons/utils/BufferGeometryUtils.js',
      'three/examples/jsm/geometries/ConvexGeometry.js',
      'three/examples/jsm/geometries/RoundedBoxGeometry.js',
      'three/examples/jsm/loaders/RGBELoader.js',
      'three/examples/jsm/math/ImprovedNoise.js',
      'three/examples/jsm/objects/Sky.js',
      'three/examples/jsm/postprocessing/OutputPass.js',
      'three/examples/jsm/postprocessing/Pass.js',
      'three/examples/jsm/shaders/CopyShader.js',
      'three/examples/jsm/utils/BufferGeometryUtils.js',
    ],
  },
  server: {
    host: '127.0.0.1', port: Number(process.env.JACKAL_VIEW_PORT || 5174), strictPort: true,
    fs: { allow: [root] },
    proxy: { '/api': { target: `http://127.0.0.1:${process.env.JACKAL_API_PORT || 8765}`, changeOrigin: true, timeout: 0 } },
  },
  build: { outDir: path.join(root, 'dist'), emptyOutDir: true },
});
