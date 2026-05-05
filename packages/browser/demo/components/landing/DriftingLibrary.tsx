import { useEffect, useRef } from "react";
import * as THREE from "three";

const PINK   = 0xF4C8D7;
const BLUE   = 0xBAD4E8;
const CREAM  = 0xF0DDB8;
const LILAC  = 0xE0D0E8;
const PEACH  = 0xF6D2BC;
const SAGE   = 0xCFE0CC;
const BOOK_COLORS = [PINK, BLUE, CREAM, LILAC, PEACH, SAGE];

const BOOK_COUNT = 120;

function makeBook(color: number): THREE.Group {
  const grp = new THREE.Group();
  const w = 0.42, h = 0.62, t = 0.13;
  const pages = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.94, h * 0.96, t * 0.78),
    new THREE.MeshStandardMaterial({ color: 0xF8F2E4, roughness: 0.95 }),
  );
  grp.add(pages);
  const cover = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, t),
    new THREE.MeshStandardMaterial({ color, roughness: 0.55 }),
  );
  grp.add(cover);
  return grp;
}

function makeGlowSprite(): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 256;
  const g = c.getContext("2d")!;
  const rg = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  rg.addColorStop(0, "rgba(97,92,237,0.30)");
  rg.addColorStop(1, "rgba(97,92,237,0)");
  g.fillStyle = rg;
  g.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(c);
  return new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      blending: THREE.AdditiveBlending,
    }),
  );
}

export function DriftingLibrary() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(dpr);
    renderer.setClearColor(0x0F0D11, 1);
    container.appendChild(renderer.domElement);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0F0D11, 9, 22);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 11);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.HemisphereLight(0xFFE8D8, 0x2A2030, 0.5));
    const key = new THREE.DirectionalLight(0xFFD8B0, 1.0);
    key.position.set(3, 4, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x8884F6, 0.35);
    rim.position.set(-4, 2, -3);
    scene.add(rim);

    const glow = makeGlowSprite();
    glow.scale.set(9, 9, 1);
    glow.position.z = -3;
    scene.add(glow);

    type BookData = {
      mesh: THREE.Group;
      a: number;
      r: number;
      y0: number;
      av: number;
      bob: number;
      rx: number;
      ry: number;
    };

    const books: BookData[] = [];
    for (let i = 0; i < BOOK_COUNT; i++) {
      const color = BOOK_COLORS[i % BOOK_COLORS.length];
      const mesh = makeBook(color);
      const a = Math.random() * Math.PI * 2;
      const r = 3.0 + Math.random() * 5.0;
      const y0 = (Math.random() - 0.5) * 7;
      mesh.position.set(Math.cos(a) * r, y0, Math.sin(a) * r - 1);
      mesh.rotation.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
      scene.add(mesh);
      books.push({
        mesh,
        a,
        r,
        y0,
        av: 0.05 + Math.random() * 0.05,
        bob: Math.random() * Math.PI * 2,
        rx: (Math.random() - 0.5) * 0.4,
        ry: (Math.random() - 0.5) * 0.4,
      });
    }

    function fit() {
      const w = container!.clientWidth;
      const h = container!.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(container);

    let frameId = 0;
    function tick(now: number) {
      const t = now * 0.001;
      for (const b of books) {
        b.a += b.av * 0.01;
        b.mesh.position.x = Math.cos(b.a) * b.r;
        b.mesh.position.z = Math.sin(b.a) * b.r - 1;
        b.mesh.position.y = b.y0 + Math.sin(t * 0.3 + b.bob) * 0.15;
        b.mesh.rotation.x += b.rx * 0.003;
        b.mesh.rotation.y += b.ry * 0.003;
      }
      renderer.render(scene, camera);
      if (!reducedMotion) {
        frameId = requestAnimationFrame(tick);
      }
    }
    if (reducedMotion) {
      tick(0); // single static frame
    } else {
      frameId = requestAnimationFrame(tick);
    }

    return () => {
      cancelAnimationFrame(frameId);
      ro.disconnect();
      // Dispose three.js resources
      for (const b of books) {
        b.mesh.traverse((o) => {
          if ((o as THREE.Mesh).geometry) {
            (o as THREE.Mesh).geometry.dispose();
          }
          const mat = (o as THREE.Mesh).material;
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else if (mat) (mat as THREE.Material).dispose();
        });
        scene.remove(b.mesh);
      }
      glow.material.map?.dispose();
      glow.material.dispose();
      scene.remove(glow);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={containerRef} className="landing-bg-canvas" />;
}
