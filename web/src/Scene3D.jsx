import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

const toWorld = (p) => new THREE.Vector3(p[0], p[2], -p[1]);

function buildScene(ep, bounds, C) {
  const [xmin, xmax, ymin, ymax] = bounds;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(C.ground);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(30, 60, 20);
  scene.add(sun);

  const size = Math.max(xmax - xmin, ymax - ymin);
  const grid = new THREE.GridHelper(size, size / 10, C.line, C.line);
  grid.position.set((xmin + xmax) / 2, 0, -(ymin + ymax) / 2);
  scene.add(grid);

  const dropMat = new THREE.LineDashedMaterial({ color: C.muted, dashSize: 0.6, gapSize: 0.6, transparent: true, opacity: 0.7 });
  const dropLine = (top) => {
    const g = new THREE.BufferGeometry().setFromPoints([top.clone(), new THREE.Vector3(top.x, 0, top.z)]);
    const l = new THREE.Line(g, dropMat);
    l.computeLineDistances();
    return l;
  };

  const obstacleMat = new THREE.MeshStandardMaterial({ color: C.lo, roughness: 0.8, transparent: true, opacity: 0.85 });
  ep.obstacles.forEach((o) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(o.radius, 32, 16), obstacleMat);
    m.position.copy(toWorld(o.pos));
    scene.add(m);
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(o.radius, 32), new THREE.MeshBasicMaterial({ color: C.lo, transparent: true, opacity: 0.25 }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(o.pos[0], 0.02, -o.pos[1]);
    scene.add(shadow);
  });

  ep.targets.forEach((t, i) => {
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(1.1), new THREE.MeshStandardMaterial({ color: C.drones[i], emissive: C.drones[i], emissiveIntensity: 0.35 }));
    m.position.copy(toWorld(t));
    scene.add(m, dropLine(m.position));
  });

  const n = ep.targets.length;
  const drones = [];
  const lineMats = [];
  for (let i = 0; i < n; i++) {
    const pts = new Float32Array(ep.frames.length * 3);
    ep.frames.forEach((f, t) => toWorld(f.pos[i]).toArray(pts, t * 3));
    const trailGeom = new LineGeometry();
    trailGeom.setPositions(pts);
    const trailMat = new LineMaterial({ color: C.drones[i], linewidth: 3 });
    lineMats.push(trailMat);
    const trail = new Line2(trailGeom, trailMat);
    const bodyMat = new THREE.MeshStandardMaterial({ color: C.drones[i], emissive: 0x000000 });
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.9, 24, 12), bodyMat);
    const drop = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), dropMat);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1.8, 0.18, 8, 32), new THREE.MeshBasicMaterial({ color: C.crash }));
    scene.add(trail, body, drop, ring);
    drones.push({ trail, body, bodyMat, drop, ring });
  }
  return { scene, drones, lineMats };
}

function update(world, ep, k, C) {
  const idx = Math.min(k, ep.frames.length - 1);
  const frame = ep.frames[idx];
  world.drones.forEach((d, i) => {
    const p = toWorld(frame.pos[i]);
    d.trail.geometry.instanceCount = idx;
    d.body.position.copy(p);
    d.bodyMat.emissive.set(frame.arrived[i] ? C.ok : 0x000000);
    d.bodyMat.emissiveIntensity = frame.arrived[i] ? 0.6 : 0;
    d.drop.geometry.setFromPoints([p, new THREE.Vector3(p.x, 0, p.z)]);
    d.drop.computeLineDistances();
    d.ring.visible = frame.hit[i];
    d.ring.position.copy(p);
    d.ring.lookAt(p.x, p.y + 10, p.z);
  });
}

export default function Scene3D({ ep, k, bounds, tokens, resetSignal }) {
  const mountRef = useRef(null);
  const ctx = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    mount.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    let raf;
    const viewport = new THREE.Vector2();
    const loop = () => {
      controls.update();
      const world = ctx.current?.world;
      if (world) {
        renderer.getSize(viewport);
        world.lineMats.forEach((m) => m.resolution.copy(viewport));
        renderer.render(world.scene, camera);
      }
      raf = requestAnimationFrame(loop);
    };
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect;
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    });
    ro.observe(mount);
    ctx.current = { renderer, camera, controls, world: null };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    const { camera, controls } = ctx.current;
    camera.position.set(0, 58, 70);
    controls.target.set(0, 3, 0);
    controls.update();
  }, [resetSignal]);

  useEffect(() => {
    const world = buildScene(ep, bounds, tokens);
    ctx.current.world = world;
    return () => world.scene.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); });
  }, [ep, bounds, tokens]);

  useEffect(() => {
    update(ctx.current.world, ep, k, tokens);
  }, [ep, k, tokens]);

  return <div className="scene3d" ref={mountRef} role="img" aria-label="3D view of the selected pilot's flight. Drag to rotate, scroll to zoom." />;
}
