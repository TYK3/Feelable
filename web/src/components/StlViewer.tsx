// 3D preview of the printable relief (lazy-loaded: keep three.js out of the first paint).
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export default function StlViewer({ url, label }: { url: string; label: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    setStatus("loading");

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setStatus("error");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
    scene.add(new THREE.HemisphereLight(0xfffbf0, 0xb8ad8f, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(-0.6, 1.0, 0.9);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xffe7a0, 0.7);
    rim.position.set(0.8, -0.4, 0.6);
    scene.add(rim);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const material = new THREE.MeshStandardMaterial({ color: 0xf6eedb, roughness: 0.62, metalness: 0.0 });
    let geometry: THREE.BufferGeometry | null = null;
    let raf = 0;
    let needs = true;

    const resize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      needs = true;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(host);
    resize();

    const loop = () => {
      raf = requestAnimationFrame(loop);
      const moved = controls.update();
      if (moved || needs) {
        renderer.render(scene, camera);
        needs = false;
      }
    };
    controls.addEventListener("change", () => (needs = true));

    new STLLoader().load(
      url,
      (geo) => {
        if (disposed) {
          geo.dispose();
          return;
        }
        geometry = geo;
        geo.computeVertexNormals();
        geo.computeBoundingBox();
        const box = geo.boundingBox!;
        const center = box.getCenter(new THREE.Vector3());
        geo.translate(-center.x, -center.y, -center.z);
        const size = box.getSize(new THREE.Vector3());
        const mesh = new THREE.Mesh(geo, material);
        scene.add(mesh);

        // Look down at the page from slightly in front, so raised lines catch the light.
        const span = Math.max(size.x, size.y);
        const vfov = (camera.fov * Math.PI) / 180;
        const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
        const dist = Math.max(size.y / 2 / Math.tan(vfov / 2), size.x / 2 / Math.tan(hfov / 2)) * 1.12;
        camera.up.set(0, 0, 1);
        camera.position.set(0, -dist * 0.62, dist * 0.82);
        camera.near = dist / 100;
        camera.far = dist * 10;
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        controls.minDistance = span * 0.15;
        controls.maxDistance = dist * 3;
        controls.update();
        needs = true;
        setStatus("ready");
      },
      undefined,
      () => !disposed && setStatus("error"),
    );
    loop();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      geometry?.dispose();
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, [url]);

  return (
    <div className="stl" role="img" aria-label={`3D model of the raised relief: ${label}. Drag to rotate, scroll to zoom.`}>
      <div className="stl__canvas" ref={hostRef} />
      {status === "loading" && (
        <div className="stl__status">
          <span className="spinner spinner--lg" aria-hidden="true" />
          Loading 3D model…
        </div>
      )}
      {status === "error" && <div className="stl__status">The 3D preview couldn't load here. The STL download still works.</div>}
      {status === "ready" && <p className="stl__hint" aria-hidden="true">Drag to rotate · Scroll or pinch to zoom</p>}
    </div>
  );
}
