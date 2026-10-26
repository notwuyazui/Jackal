import * as THREE from 'three';
import { createRenderer, onResize } from '@cot/engine/renderer.ts';
import { createCameraRig } from '@cot/engine/cameraRig.ts';
import { createSky } from '@cot/engine/sky.ts';
import { createColliders } from './colliders.ts';
import { createShellPresentation } from './shells.ts';
import { LATE_FX_LAYER } from '@cot/fx/layers.ts';
import { copyResolvedDepth } from '@cot/engine/resolvedDepthCopy.ts';
import { LateFxSceneView } from '@cot/engine/lateFxSceneView.ts';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { toWorld, toYaw, toHeading } from './mapping.ts';
import type { Frame, Parameters, UnitFrame, RenderReceipt } from './protocol.ts';
import type { GroundDisturbance } from '@cot/world/groundPressure.ts';
type MutableDisturbance = { -readonly [K in keyof GroundDisturbance]: GroundDisturbance[K] };

import { prepareMap, type InitialPlacement, type MapScene, type PreparedMap } from '../maps/registry.ts';
import { createVehicle, prepareVehicles, type Vehicle, TEAM_CAMO } from './models.ts';

export async function createView(container: HTMLElement) {
  const labelPoint = new THREE.Vector3();
  const cameraPosition = new THREE.Vector3();
  const cameraTarget = new THREE.Vector3();
  const cameraDirection = new THREE.Vector3();
  const effectPosition = new THREE.Vector3();
  const effectDirection = new THREE.Vector3();
  const disturbances: GroundDisturbance[] = [];
  const disturbanceByUnit = new Map<number, MutableDisturbance>();
  const captureAges = [0.072, 0.24, 0.6];
  const captureNames = ['muzzle-peak', 'muzzle-recovery', 'muzzle-smoke'];
  // This bridge renders directly through Three.js. CoT's volumetric clouds
  // require its post-processing compositor, so use CoT's shaded baked deck.
  const cloudUrl = new URL(window.location.href);
  if (cloudUrl.searchParams.get('clouds') !== 'baked') {
    cloudUrl.searchParams.set('clouds', 'baked');
    window.history.replaceState(null, '', cloudUrl);
  }
  const loadingStage = (text: string) => { document.getElementById("loading")!.textContent = text; };
  loadingStage("正在加载天空与云层…");
  const renderer = createRenderer(container);
  renderer.domElement.setAttribute('aria-label', 'Claude of Tanks M1A1 同步 3D 场景');
  const maxPixelRatio = Math.min(window.devicePixelRatio || 1, 1.25);
  let displayPixelRatio = maxPixelRatio;
  renderer.setPixelRatio(displayPixelRatio);
  renderer.toneMappingExposure = 1.10;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(58, 1, 0.25, 16000);
  const sky = createSky(scene, renderer);
  await sky.ensureCloudTexturesChunked();
  sky.applyPreset({ sunElevationDeg: 32, sunAzimuthDeg: 115, skyIntensity: 1.24,
    cloudOpacity: 1.02, cloudOpacity2: 0.48, cloudAltM: 1400,
    cloudHazeK: 0.00008, cloudUvM: 2600, cloudTintHex: 0xffffff,
    envIntensity: 0.25,
    fogDensity: 0.00038, fogTintHex: 0xa5bed0, fogMix: 0.48 }, scene);
  const cumulus = scene.getObjectByName('cloudLayer') as THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial> | undefined;
  const rig = createCameraRig(camera, {
    heightField: { getHeightAt: () => 0 }, raycast: () => null, getPlayer: () => null,
  });
  const hemisphere = new THREE.HemisphereLight(0xe6f2ff, 0x968966, 1.1);
  const sun = new THREE.DirectionalLight(0xffefd9, 2.0);
  sun.position.copy(sky.sunDir).multiplyScalar(180);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -145, right: 145, top: 105, bottom: -105, near: 1, far: 320 });
  sun.shadow.bias = -0.00015;
  sun.shadow.normalBias = 0.025;
  sun.shadow.camera.updateProjectionMatrix();
  const fill = new THREE.DirectionalLight(0xe0e9df, 0.5);
  fill.position.copy(sky.sunDir).multiplyScalar(-120);
  fill.position.y = 75;
  const chaseFill = new THREE.PointLight(0xf0f1dd, 65, 24, 2);
  chaseFill.castShadow = false;
  scene.add(hemisphere, sun, sun.target, fill, chaseFill);
  const world = new THREE.Group();
  world.name = 'Jackal authored scene';
  scene.add(world);
  const labels = document.getElementById('labels')!;
  const models = new Map<number, Vehicle>();
  const colliders = new Map<number, ReturnType<typeof createColliders>>();
  const labelElements = new Map<number, HTMLDivElement>();
  const unitsById = new Map<number, UnitFrame>();
  const parametersById = new Map<number, Parameters>();
  const placementsById = new Map<number, InitialPlacement>();
  function required<T>(map: ReadonlyMap<number, T>, id: number, label: string): T {
    const value = map.get(id);
    if (value === undefined) throw new Error(`${label}缺少单位 #${id}；变更单位列表后请重新 initialize(manifest)`);
    return value;
  }
  const vehicle = (id: number) => required(models, id, '模型');
  const previousAlive = new Map<number, boolean>();
  loadingStage("正在加载战斗特效…");
  const shells = await createShellPresentation(scene, camera);
  // Native CoT smoke/flash shaders require a resolved world depth source.
  // The source is an offscreen target, never the framebuffer they draw into.
  const worldDepth = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true, type: THREE.HalfFloatType,
    depthTexture: new THREE.DepthTexture(1, 1, THREE.UnsignedIntType) });
  const composite = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true, type: THREE.HalfFloatType,
    depthTexture: new THREE.DepthTexture(1, 1, THREE.UnsignedIntType) });
  const copyMaterial = new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.clone(CopyShader.uniforms),
    vertexShader: CopyShader.vertexShader, fragmentShader: CopyShader.fragmentShader,
    depthTest: false, depthWrite: false, blending: THREE.NoBlending, toneMapped: false });
  const copyQuad = new FullScreenQuad(copyMaterial);
  const output = new OutputPass();
  output.renderToScreen = true;
  const fxView = new LateFxSceneView(scene);
  const bufferSize = new THREE.Vector2();
  const debugColliders = new URLSearchParams(window.location.search).get('colliders') === '1';
  const viewToken = new URLSearchParams(window.location.search).get('token');
  let battlePlaneHeight = NaN;
  let currentRun = -1;
  let currentFrame: Frame | null = null;
  let mapScene: MapScene | null = null;
  let mapFactory: PreparedMap | null = null;
  let mapBuilder: string | null = null;
  let grassTick = -1;
  let grassWarm = false;
  let lastTick = -1;
  let firstContactCaptured = false;
  let firstMuzzleCaptured = false;
  let fireCount = 0;
  let destructionCount = 0;
  let visualClock = 0;
  let lastVisualWallMs = performance.now();
  let visualRunning = false;
  let disposed = false;
  let raf = 0;
  let animationFrames = 0;
  let presented: (() => void) | null = null;
  const frameTimes: number[] = [];
  const renderCosts: number[] = [];
  let peakDrawCalls = 0, peakTriangles = 0;
  let lastShadowTick = -1, lastResolutionCheck = 0, resolutionChanges = 0;
  let shadowDirty = false;
  const shotsByUnit: Record<number, number> = {};
  const animationMaxima = { recoil_m: 0, pitch_deg: 0, roll_deg: 0 };
  const pendingEffectCaptures: Array<{ name: string; png: string; age_s: number;
    recoil_m: number; pitch_deg: number; flash_particles: number; smoke_particles: number }> = [];
  let shotCapture: { born: number; source: number; stage: number } | null = null;


  function resize() {
    if (disposed) return;
    onResize(renderer, camera);
    renderer.setPixelRatio(displayPixelRatio);
    renderer.getDrawingBufferSize(bufferSize);
    worldDepth.setSize(bufferSize.x, bufferSize.y);
    composite.setSize(bufferSize.x, bufferSize.y);
    shells.bindSceneDepth(worldDepth.depthTexture!, bufferSize.x, bufferSize.y);
    if (currentFrame && !visualRunning) renderScene(currentFrame);
  }
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  renderer.userData.contextRecovery = {
    onLost() { container.dataset.connection = '图形设备恢复中'; },
    onRestored() { resize(); return true; },
  };

  async function reset(frame: Frame) {
    mapScene?.dispose(); mapScene = null;
    if (mapBuilder !== frame.manifest.scene.builder) {
      mapFactory?.dispose(); mapFactory = null;
      const prepared = await prepareMap(frame.manifest.scene, { renderer, loadingStage });
      if (disposed) { prepared.dispose(); throw new Error('3D 视图已关闭'); }
      mapFactory = prepared;
      mapBuilder = frame.manifest.scene.builder;
    }
    await prepareVehicles(frame.manifest.scene);
    grassTick = -1;
    disturbances.length = 0;
    disturbanceByUnit.clear();
    grassWarm = false;
    for (const model of models.values()) model.dispose();
    models.clear();
    for (const collider of colliders.values()) collider.dispose();
    colliders.clear();
    labels.replaceChildren();
    labelElements.clear();
    previousAlive.clear();
    parametersById.clear();
    placementsById.clear();
    for (const pose of frame.manifest.scene.initial_units ?? []) placementsById.set(pose.id, pose);
    for (const parameters of frame.manifest.units) parametersById.set(parameters.id, parameters);
    fireCount = 0;
    destructionCount = 0;
    visualClock = 0;
    lastVisualWallMs = performance.now();
    animationFrames = 0;
    frameTimes.length = renderCosts.length = 0;
    peakDrawCalls = peakTriangles = 0;
    lastShadowTick = -1;
    lastResolutionCheck = resolutionChanges = 0;
    for (const id of Object.keys(shotsByUnit)) delete shotsByUnit[Number(id)];
    animationMaxima.recoil_m = animationMaxima.pitch_deg = animationMaxima.roll_deg = 0;
    shotCapture = null;
    pendingEffectCaptures.length = 0;
    shells.reset(frame.manifest.seed);
    battlePlaneHeight = NaN;
    world.clear();
    const nextScene = await mapFactory!.build(frame);
    if (disposed) { nextScene.dispose(); throw new Error('3D 视图已关闭'); }
    mapScene = nextScene;
    world.add(mapScene.group);
    firstContactCaptured = false;
    firstMuzzleCaptured = false;
    lastTick = -1;
  }

  function renderScene(frame: Frame) {
    const renderStarted = performance.now();
    renderer.info.autoReset = false;
    renderer.info.reset();
    // The asynchronous CoT cloud bake refreshes uniforms when its worker
    // finishes, so apply this display-only diffuse tint on each rendered frame.
    if (cumulus) {
      (cumulus.material.uniforms.uTint.value as THREE.Color).setRGB(1.55, 1.55, 1.48);
      cumulus.material.uniforms.uShadeW.value = 0.34;
    }
    const followed = required(unitsById, frame.manifest.follow_id, '跟随对象');
    const position = vehicle(followed.id).root.position;
    const yaw = toYaw(followed.turret_heading);
    // The chase camera stays behind the gun and follows the turret bearing.
    // The fixed entity remains #2, including after it has been destroyed.
    cameraDirection.set(Math.sin(yaw), 0, Math.cos(yaw));
    vehicle(followed.id).cameraPivot(cameraTarget);
    cameraPosition.copy(cameraTarget).addScaledVector(cameraDirection, -13 * Math.cos(THREE.MathUtils.degToRad(10)));
    cameraPosition.y += 13 * Math.sin(THREE.MathUtils.degToRad(10));
    chaseFill.position.copy(cameraPosition);
    rig.setExternalPose(cameraPosition, cameraTarget, 60);
    const grass = mapScene?.grass;
    if (grass) {
      const scale = frame.manifest.meters_per_pixel;
      if (grassTick !== frame.tick) {
        disturbances.length = 0;
        for (const unit of frame.units) {
          if (!unit.alive) continue;
          let disturbance = disturbanceByUnit.get(unit.id);
          if (!disturbance) {
            disturbance = { x: 0, z: 0, dirX: 0, dirZ: 0, speed: 0, halfWidth: 0, halfLength: 0 };
            disturbanceByUnit.set(unit.id, disturbance);
          }
          const p = vehicle(unit.id).root.position;
          const facing = toYaw(unit.heading), travelSign = unit.speed < 0 ? -1 : 1;
          disturbance.x = p.x; disturbance.z = p.z;
          disturbance.dirX = Math.sin(facing) * travelSign;
          disturbance.dirZ = Math.cos(facing) * travelSign;
          disturbance.speed = Math.abs(unit.speed) * scale;
          disturbance.halfWidth = unit.size[0] * scale / 2;
          disturbance.halfLength = unit.size[1] * scale / 2;
          disturbances.push(disturbance);
        }
        grass.setDisturbances(disturbances);
      }
      grass.update(grassTick === frame.tick || grassTick < 0 ? 0 : frame.manifest.dt,
        cameraPosition, position, cameraDirection);
      if (!grassWarm) {
        for (let i = 0; i < 30; i++) {
          const state = grass.getState();
          if (!state.near.pending && !state.far.pending) break;
          grass.update(0, cameraPosition, position, cameraDirection);
        }
        grassWarm = true;
      }
      grass.setWindTime(frame.time);
      grassTick = frame.tick;
    }
    camera.updateMatrixWorld(true);
    mapScene?.exterior?.setTime(frame.time);
    shells.render();
    const layers = camera.layers.mask;
    const background = scene.background;
    const matrixAuto = scene.matrixWorldAutoUpdate;
    const autoClear = renderer.autoClear;
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = frame.tick !== lastShadowTick || shadowDirty;
    shadowDirty = false;
    lastShadowTick = frame.tick;
    camera.layers.disable(LATE_FX_LAYER);
    renderer.setRenderTarget(worldDepth);
    renderer.render(scene, camera);
    // Reuse the world color and resolved depth. Only native late particles and
    // tracers draw again, preserving soft intersections and hardware occlusion.
    renderer.autoClear = false;
    renderer.setRenderTarget(composite);
    renderer.clear(true, true, false);
    copyMaterial.uniforms.tDiffuse.value = worldDepth.texture;
    copyQuad.render(renderer);
    copyResolvedDepth(renderer, worldDepth, composite);
    renderer.setRenderTarget(composite);
    camera.layers.set(LATE_FX_LAYER);
    scene.background = null;
    scene.matrixWorldAutoUpdate = false;
    renderer.render(fxView.select(camera), camera);
    scene.matrixWorldAutoUpdate = matrixAuto;
    scene.background = background;
    camera.layers.mask = layers;
    output.render(renderer, composite, composite, 0, false);
    renderer.autoClear = autoClear;
    peakDrawCalls = Math.max(peakDrawCalls, renderer.info.render.calls);
    peakTriangles = Math.max(peakTriangles, renderer.info.render.triangles);
    if (visualClock > 2) {
      renderCosts.push(performance.now() - renderStarted);
      if (renderCosts.length > 300) renderCosts.shift();
    }
    const rect = renderer.domElement.getBoundingClientRect();
    for (const [id, model] of models) {
      const element = required(labelElements, id, '标签');
      model.labelAnchor(labelPoint).project(camera);
      element.hidden = !model.root.visible || labelPoint.z < -1 || labelPoint.z > 1 ||
        Math.abs(labelPoint.x) > 1 || Math.abs(labelPoint.y) > 1;
      element.style.opacity = String(model.root.visible ?
        Math.max(0, Math.min(1, Number(element.dataset.opacity ?? 1))) : 0);
      element.style.left = (labelPoint.x * 0.5 + 0.5) * rect.width + 'px';
      element.style.top = (-labelPoint.y * 0.5 + 0.5) * rect.height + 'px';
    }
  }

  function animate(now: number) {
    if (disposed) return;
    if (currentFrame && visualRunning) {
      if (visualClock > 2) {
        frameTimes.push(now - lastVisualWallMs);
        if (frameTimes.length > 300) frameTimes.shift();
      }
      visualClock += Math.min(0.05, Math.max(0, (now - lastVisualWallMs) / 1000));
      shells.pinTime(visualClock);
      for (const unit of currentFrame.units) {
        const model = vehicle(unit.id), wasVisible = model.root.visible;
        const state = model.pose(unit, visualClock);
        if (wasVisible !== model.root.visible) shadowDirty = true;
        animationMaxima.recoil_m = Math.max(animationMaxima.recoil_m, state.native_recoil_m ?? 0);
        animationMaxima.pitch_deg = Math.max(animationMaxima.pitch_deg, Math.abs(state.cosmetic_pitch_deg ?? 0));
        animationMaxima.roll_deg = Math.max(animationMaxima.roll_deg, Math.abs(state.chassis_roll_deg ?? 0));
        required(labelElements, unit.id, '标签').dataset.opacity = String(state.visual_opacity ?? 1);
      }
      renderScene(currentFrame);
      animationFrames++;
      if (visualClock - lastResolutionCheck > 4 && frameTimes.length >= 90) {
        const fps = performanceReceipt().average_fps;
        const next = fps < 48 ? Math.max(.75, displayPixelRatio - .15) :
          fps > 59 ? Math.min(maxPixelRatio, displayPixelRatio + .05) : displayPixelRatio;
        lastResolutionCheck = visualClock;
        if (Math.abs(next - displayPixelRatio) > .01) {
          displayPixelRatio = next;
          resolutionChanges++;
          resize();
        }
      }
      if (shotCapture) {
        const age = visualClock - shotCapture.born;
        if (age >= captureAges[shotCapture.stage]) {
          const state = vehicle(shotCapture.source).presentationState();
          if (!state) throw new Error("开火模型尚未接收姿态");
          pendingEffectCaptures.push({ name: captureNames[shotCapture.stage], age_s: age,
            recoil_m: state.native_recoil_m ?? 0, pitch_deg: state.cosmetic_pitch_deg ?? 0,
            ...shells.effectState(), png: renderer.domElement.toDataURL('image/png') });
          shotCapture.stage++;
          if (shotCapture.stage === captureAges.length) shotCapture = null;
        }
      }
      const ready = presented;
      presented = null;
      ready?.();
    }
    lastVisualWallMs = now;
    raf = requestAnimationFrame(animate);
  }

  function performanceReceipt() {
    const sorted = frameTimes.slice().sort((a, b) => a - b);
    const mean = frameTimes.reduce((sum, value) => sum + value, 0) / (frameTimes.length || 1);
    return { samples: frameTimes.length, average_fps: mean > 0 ? 1000 / mean : 0,
      p95_frame_ms: sorted.length ? sorted[Math.floor((sorted.length - 1) * .95)] : 0,
      mean_render_cpu_ms: renderCosts.reduce((sum, value) => sum + value, 0) / (renderCosts.length || 1),
      peak_draw_calls: peakDrawCalls, peak_triangles: peakTriangles,
      gpu_geometries: renderer.info.memory.geometries, gpu_textures: renderer.info.memory.textures,
      pixel_ratio: renderer.getPixelRatio(), world_passes_per_frame: 1,
      resolution_changes: resolutionChanges };
  }

  async function draw(frame: Frame, onPresented: () => void) {
    if (currentRun !== frame.run_id) {
      visualRunning = false;
      currentFrame = null;
      loadingStage("正在生成远近山体…");
      await reset(frame);
      currentRun = frame.run_id;
    }
    unitsById.clear();
    for (const unit of frame.units) unitsById.set(unit.id, unit);
    required(unitsById, frame.manifest.follow_id, "跟随对象");
    shells.pinTime(visualClock);
    const rendered: RenderReceipt[] = [];
    for (const unit of frame.units) {
      let model = models.get(unit.id);
      if (!model) {
        const parameters = required(parametersById, unit.id, '模型参数');
        model = createVehicle(parameters, unit.team, frame.manifest);
        const placement = placementsById.get(unit.id);
        if (placement) {
          model.root.position.set(placement.x, placement.y, placement.z);
          model.root.rotation.y = placement.hull_yaw;
        }
        models.set(unit.id, model);
        scene.add(model.root);
        const element = document.createElement('div');
        element.className = 'tank-label' + (unit.team === 'ENEMY' ? ' enemy' : '');
        labels.appendChild(element);
        labelElements.set(unit.id, element);
      }
      if (frame.tick !== lastTick && frame.events.some(e => e.kind === 'fire' && e.source_id === unit.id)) model.fire(visualClock);
      const receipt = model.pose(unit, visualClock, frame.tick !== lastTick && frame.tick > 0 ? frame.manifest.dt : 0);
      rendered.push(receipt);
      if (frame.tick !== lastTick && previousAlive.get(unit.id) === true && !unit.alive) {
        effectPosition.copy(model.root.position);
        effectPosition.y = model.modelHeight * 0.48;
        shells.destruction(effectPosition);
        destructionCount++;
      }
      previousAlive.set(unit.id, unit.alive);
      required(labelElements, unit.id, '标签').innerHTML = '#' + unit.id + ' · ' + (unit.alive ? unit.health : '击毁') +
        '<i><b style="width:' + Math.max(0, unit.health / unit.max_health * 100) + '%"></b></i>';
      required(labelElements, unit.id, '标签').dataset.opacity = String(receipt.visual_opacity ?? 1);
    }
    if (!Number.isFinite(battlePlaneHeight)) battlePlaneHeight = vehicle(frame.manifest.follow_id).muzzleHeight;
    for (let index = 0; index < frame.units.length; index++) {
      const unit = frame.units[index];
      let collider = colliders.get(unit.id);
      if (!collider) {
        collider = createColliders(frame.manifest,
          Math.max(vehicle(unit.id).modelHeight, battlePlaneHeight + 0.2), debugColliders);
        colliders.set(unit.id, collider);
        // Separate scene branch: the visual hull bearing cannot rotate these rectangles.
        if (unit.alive) scene.add(collider.group);
      }
      if (unit.alive && !collider.group.parent) scene.add(collider.group);
      if (!unit.alive) collider.group.removeFromParent();
      Object.assign(rendered[index], collider.pose(unit));
    }
    if (frame.tick !== lastTick) {
      for (const event of frame.events) {
        if (event.kind === 'fire' && event.source_id != null) {
          const unit = required(unitsById, event.source_id, '开火来源');
          vehicle(unit.id).muzzleWorld(effectPosition);
          const yaw = toYaw(unit.turret_heading);
          effectDirection.set(Math.sin(yaw), 0, Math.cos(yaw));
          shells.muzzleFlash(effectPosition, effectDirection);
          fireCount++;
          shotsByUnit[unit.id] = (shotsByUnit[unit.id] || 0) + 1;
          if (!firstMuzzleCaptured && unit.id === frame.manifest.follow_id) {
            firstMuzzleCaptured = true;
            shotCapture = { born: visualClock, source: unit.id, stage: 0 };
          }
        } else if (event.kind === 'impact' && event.position) {
          const point = toWorld(...event.position, ...frame.manifest.map.size, frame.manifest.meters_per_pixel);
          effectPosition.set(point.x, battlePlaneHeight, point.z);
          shells.impact(event.bullet_id ?? undefined, effectPosition);
        }
      }
    }
    shells.sync(frame, battlePlaneHeight);
    mapScene?.roadside?.sync(frame);
    currentFrame = frame;
    visualRunning = true;
    lastTick = frame.tick;
    // One RAF owns display rendering. State ingestion never draws the world
    // a second time or waits for an additional frame after the display is ready.
    if (frame.tick === 0) loadingStage("正在呈现首帧…");
    await new Promise<void>(resolve => { presented = resolve; });
    // Publish the exact 2D/HUD state immediately, even when a final wreck
    // animation continues before the final verification receipt is returned.
    onPresented();
    if (frame.result.finished && frame.tick !== 0) {
      const finishAt = visualClock + 3.65;
      await new Promise<void>(resolve => {
        const settled = () => {
          if (visualClock < finishAt && !disposed) requestAnimationFrame(settled);
          else resolve();
        };
        requestAnimationFrame(settled);
      });
      visualRunning = false;
      for (let i = 0; i < frame.units.length; i++) {
        const state = vehicle(frame.units[i].id).presentationState();
        if (!state) throw new Error('结算模型尚未接收姿态');
        for (const key of ['visual_opacity', 'wreck_age_s', 'recoil_age_s', 'visible',
          'cosmetic_bob_m', 'cosmetic_pitch_deg', 'chassis_roll_deg', 'cosmetic_retreat_m',
          'native_recoil_m', 'start_bounce_age_s'] as const) Object.assign(rendered[i], { [key]: state[key] });
      }
    }
    const contact = !firstContactCaptured && frame.events.some(e => e.kind === 'impact');
    if (contact) firstContactCaptured = true;
    const screenshot = frame.tick === 0 || contact || frame.result.finished ? renderer.domElement.toDataURL('image/png') : undefined;
    const effect_captures = pendingEffectCaptures.splice(0);
    camera.getWorldDirection(cameraDirection);
    return { view_token: viewToken, rendered, bullets: shells.receipts(frame.manifest),
      performance: performanceReceipt(),
      scenery: { map_id: frame.manifest.scene.id, map_builder: frame.manifest.scene.builder, scene_revision: 'verdant-native-continuous-v7',
        model: 'cot-native-m1a1', ground: 'cot-grass004', grass: mapScene?.grass?.getState(),
        sky: 'cot-baked-cumulus-verdant', lighting: 'diffuse-soil-sun-entity-shadows',
        shell: 'cot-native-he-tracer', shell_textures: 'cot-original-particle-atlases',
        battle_plane_height_m: battlePlaneHeight, collider: 'jackal-axis-aligned-cuboid',
        boundary: 'transparent-air-barrier', exterior: mapScene?.exterior?.counts,
        ground_dressing: mapScene?.groundDressing?.counts, roadside: mapScene?.roadside &&
          { ...mapScene.roadside.counts, crushed: mapScene.roadside.crushed }, camouflage: TEAM_CAMO,
        effects: { muzzle_flashes: fireCount, destructions: destructionCount,
          muzzle_style: 'cot-native-continuous-200mm-visual',
          animation_frames: animationFrames, animation_maxima: { ...animationMaxima },
          shots_by_unit: { ...shotsByUnit }, soft_depth_bound: shells.effectState().soft_depth_bound,
          wreck_fade: 'cot-burn-then-alpha-3.55s' } },
      camera: { follow_id: frame.manifest.follow_id, mode: 'turret-follow',
        heading: toHeading(Math.atan2(cameraDirection.x, cameraDirection.z)),
        orbit_distance_m: cameraPosition.distanceTo(cameraTarget),
        orbit_pitch_deg: THREE.MathUtils.radToDeg(Math.atan2(cameraPosition.y - cameraTarget.y,
          Math.hypot(cameraPosition.x - cameraTarget.x, cameraPosition.z - cameraTarget.z))),
        fov_deg: camera.fov,
        anchor_x: vehicle(frame.manifest.follow_id).root.position.x / frame.manifest.meters_per_pixel + frame.manifest.map.size[0] / 2,
        anchor_y: vehicle(frame.manifest.follow_id).root.position.z / frame.manifest.meters_per_pixel + frame.manifest.map.size[1] / 2 },
      screenshot, effect_captures };
  }
  loadingStage("正在连接战斗画面…");
  resize();
  raf = requestAnimationFrame(animate);
  return {
    draw,
    dispose() {
      if (disposed) return;
      disposed = true;
      cancelAnimationFrame(raf);
      presented?.(); presented = null;
      mapScene?.dispose(); mapFactory?.dispose();
      observer.disconnect(); rig.release(); shells.dispose();
      worldDepth.dispose(); composite.dispose(); output.dispose(); copyQuad.dispose(); copyMaterial.dispose();
      for (const model of models.values()) model.dispose();
      for (const collider of colliders.values()) collider.dispose();
      world.clear(); sun.shadow.map?.dispose(); renderer.dispose();
    },
  };
}
