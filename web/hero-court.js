"use strict";

(() => {
  const hero = document.getElementById("hero-court");
  const court = document.getElementById("court");
  if (!hero || !court) return;

  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const desktopPointer = matchMedia("(min-width: 651px)");
  const limits = { minX: 0.12, maxX: 0.88, minY: 0.04, maxY: 0.96 };
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const percent = (value) => `${(value * 100).toFixed(2)}%`;

  const players = {
    north: { el: court.querySelector(".court-dot.north"), x: 0.5, y: 0.12, tx: 0.5, ty: 0.12, phase: 0, side: -1 },
    south: { el: court.querySelector(".court-dot.south"), x: 0.5, y: 0.88, tx: 0.5, ty: 0.88, phase: 2, side: 1 },
  };
  const ball = {
    el: court.querySelector(".court-ball"),
    shadow: court.querySelector(".court-shadow"),
    reticle: court.querySelector(".court-reticle"),
    x: 0.5, y: 0.88, h: 0,
  };
  const state = {
    flight: null,
    aim: { x: 0.5, y: 0.5 },
    aimVisible: false,
    hovering: false,
    pointer: { x: 0, y: 0 },
    tilt: { x: 0, y: 0 },
    lastInput: -Infinity,
    nextRally: performance.now() + 1200,
  };

  const plane = (offsetX, offsetY) => ({
    x: clamp(offsetX / court.offsetWidth, limits.minX, limits.maxX),
    y: clamp(offsetY / court.offsetHeight, limits.minY, limits.maxY),
  });

  const NET = 0.5, NET_GAP = 0.05;
  const ownerOf = (y) => (y < NET ? players.north : players.south);
  const opponentOf = (player) => (player === players.north ? players.south : players.north);
  const settleIn = (player, y) => {
    const half = player === players.north ? [limits.minY, NET - NET_GAP] : [NET + NET_GAP, limits.maxY];
    return clamp(y, half[0], half[1]);
  };
  // Un golpe siempre cruza la red: el destino se lleva a la mitad rival.
  const landingFor = (point) => ({ x: point.x, y: settleIn(opponentOf(ownerOf(ball.y)), point.y) });

  function shoot(point, now = performance.now()) {
    if (state.flight) return;
    const from = { x: ball.x, y: ball.y, h: ball.h };
    const target = landingFor(point);
    const hitter = ownerOf(from.y);
    const receiver = opponentOf(hitter);
    const still = reduceMotion.matches;

    hitter.tx = clamp(from.x, limits.minX, limits.maxX);
    hitter.ty = settleIn(hitter, from.y + 0.07 * hitter.side);
    receiver.tx = target.x;
    receiver.ty = settleIn(receiver, target.y + 0.04 * receiver.side);

    const distance = Math.hypot((target.x - from.x) * court.offsetWidth, (target.y - from.y) * court.offsetHeight);
    state.flight = {
      from,
      to: target,
      start: now,
      duration: still ? 0 : clamp(distance * 2.2 + 350, 450, 1250),
      peak: still ? 0 : clamp(distance * 0.12, 6, 28),
    };
  }

  function land(target) {
    const ring = document.createElement("i");
    ring.className = "court-ripple";
    ring.style.left = percent(target.x);
    ring.style.top = percent(target.y);
    ring.addEventListener("animationend", () => ring.remove());
    court.append(ring);
    state.nextRally = performance.now() + 900 + Math.random() * 900;
  }

  function stepBall(now) {
    const flight = state.flight;
    if (!flight) return;
    const t = flight.duration ? clamp((now - flight.start) / flight.duration, 0, 1) : 1;
    ball.x = flight.from.x + (flight.to.x - flight.from.x) * t;
    ball.y = flight.from.y + (flight.to.y - flight.from.y) * t;
    ball.h = flight.from.h * (1 - t) + 4 * flight.peak * t * (1 - t);
    if (t >= 1) {
      state.flight = null;
      ball.x = flight.to.x;
      ball.y = flight.to.y;
      ball.h = 0;
      land(flight.to);
    }
  }

  function stepPlayers(dt, now) {
    const k = 1 - Math.exp(-dt * 6);
    for (const player of Object.values(players)) {
      player.x += (player.tx - player.x) * k;
      player.y += (player.ty - player.y) * k;
      const sway = reduceMotion.matches ? 0 : Math.sin(now / 900 + player.phase) * 0.012;
      player.el.style.left = percent(player.x + sway);
      player.el.style.top = percent(player.y);
    }
  }

  function renderBall() {
    ball.el.style.left = percent(ball.x);
    ball.el.style.top = percent(ball.y);
    ball.el.style.transform = `translate(-50%, calc(-50% - ${ball.h.toFixed(1)}px))`;
    ball.shadow.style.left = percent(ball.x);
    ball.shadow.style.top = percent(ball.y);
    ball.shadow.style.opacity = clamp(1 - ball.h / 40, 0.3, 1).toFixed(2);
    ball.shadow.style.transform = `translate(-50%, -50%) scale(${clamp(1 - ball.h / 60, 0.6, 1).toFixed(2)})`;
  }

  function renderReticle() {
    const spot = landingFor(state.aim);
    ball.reticle.style.left = percent(spot.x);
    ball.reticle.style.top = percent(spot.y);
    ball.reticle.classList.toggle("on", state.aimVisible && !state.flight);
  }

  function stepTilt(dt) {
    const target = state.hovering && !reduceMotion.matches ? state.pointer : { x: 0, y: 0 };
    const k = 1 - Math.exp(-dt * 5);
    state.tilt.x += (target.x - state.tilt.x) * k;
    state.tilt.y += (target.y - state.tilt.y) * k;
    court.style.setProperty("--rz", `${(18 + state.tilt.x).toFixed(2)}deg`);
    court.style.setProperty("--rx", `${(14 + state.tilt.y).toFixed(2)}deg`);
  }

  function randomTarget() {
    const towardNorth = ball.y >= 0.5;
    return {
      x: 0.2 + Math.random() * 0.6,
      y: towardNorth ? 0.1 + Math.random() * 0.32 : 0.58 + Math.random() * 0.32,
    };
  }

  let rafId = 0;
  let last = performance.now();
  let visible = false;

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    stepBall(now);
    if (!state.flight && !reduceMotion.matches && now >= state.nextRally && now - state.lastInput > 2500) {
      shoot(randomTarget(), now);
    }
    stepPlayers(dt, now);
    renderBall();
    stepTilt(dt);
    renderReticle();
    rafId = visible ? requestAnimationFrame(frame) : 0;
  }

  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible && !rafId) {
      last = performance.now();
      rafId = requestAnimationFrame(frame);
    }
  }).observe(hero);

  const syncTabIndex = () => { court.tabIndex = desktopPointer.matches ? 0 : -1; };
  syncTabIndex();
  desktopPointer.addEventListener("change", syncTabIndex);

  hero.addEventListener("pointermove", (event) => {
    const rect = hero.getBoundingClientRect();
    const nx = clamp(((event.clientX - rect.left) / rect.width) * 2 - 1, -1, 1);
    const ny = clamp(((event.clientY - rect.top) / rect.height) * 2 - 1, -1, 1);
    state.pointer = { x: nx * 6, y: ny * -4 };
    state.hovering = true;
  });
  hero.addEventListener("pointerleave", () => { state.hovering = false; });

  court.addEventListener("pointermove", (event) => {
    state.aim = plane(event.offsetX, event.offsetY);
    state.aimVisible = true;
  });
  court.addEventListener("pointerleave", () => {
    if (document.activeElement !== court) state.aimVisible = false;
  });
  court.addEventListener("click", (event) => {
    const target = plane(event.offsetX, event.offsetY);
    state.aim = target;
    state.lastInput = performance.now();
    shoot(target);
  });

  const keySteps = { ArrowLeft: [-0.05, 0], ArrowRight: [0.05, 0], ArrowUp: [0, -0.05], ArrowDown: [0, 0.05] };
  court.addEventListener("keydown", (event) => {
    const step = keySteps[event.key];
    if (step) {
      event.preventDefault();
      state.aim = {
        x: clamp(state.aim.x + step[0], limits.minX, limits.maxX),
        y: clamp(state.aim.y + step[1], limits.minY, limits.maxY),
      };
      state.aimVisible = true;
      state.lastInput = performance.now();
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      state.lastInput = performance.now();
      shoot({ ...state.aim });
    }
  });
  court.addEventListener("focus", () => { state.aimVisible = true; });
  court.addEventListener("blur", () => { state.aimVisible = false; });
})();
