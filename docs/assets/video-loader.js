"use strict";

window.TennisVideoLoader = class {
  constructor(video) {
    this.video = video;
    this.task = null;
    this.objectURL = null;
    this.position = 0;
    this.frame = document.createElement("div");
    this.frame.className = "video-loader";
    video.replaceWith(this.frame);
    this.panel = document.createElement("div");
    this.panel.className = "video-loading-panel";
    this.status = document.createElement("p");
    this.status.setAttribute("role", "status");
    this.progress = document.createElement("progress");
    this.progress.max = 100;
    this.progress.setAttribute("aria-label", "Descarga del vídeo");
    this.failure = document.createElement("p");
    this.failure.setAttribute("role", "alert");
    this.retry = document.createElement("button");
    this.retry.type = "button";
    this.retry.textContent = "Reintentar";
    this.download = document.createElement("a");
    this.download.textContent = "Abrir MP4";
    this.download.target = "_blank";
    this.download.rel = "noopener";
    this.panel.append(this.status, this.progress, this.failure, this.retry, this.download);
    this.frame.append(video, this.panel);
    this.retry.addEventListener("click", () => { void this.load(this.source, { time: this.position }); });
    video.addEventListener("error", () => {
      if (!this.ready) return;
      console.error("No se ha podido continuar la reproducción del vídeo.", video.error);
      video.pause();
      this.position = video.currentTime;
      this.setState("error", "No se ha podido continuar la reproducción del vídeo. Reintenta o abre el MP4.");
    });
    this.setState("idle");
  }

  get ready() { return this.frame.dataset.state === "ready"; }
  get time() { return this.ready ? this.video.currentTime : this.position; }

  setState(state, message = "") {
    this.frame.dataset.state = state;
    this.frame.setAttribute("aria-busy", String(state === "loading"));
    this.video.hidden = state !== "ready";
    this.video.controls = state === "ready";
    this.panel.hidden = state === "ready";
    this.status.textContent = state === "loading" ? "Cargando vídeo…" : "Vídeo pendiente de carga.";
    this.status.hidden = state === "error";
    this.progress.hidden = state !== "loading";
    this.failure.textContent = message;
    this.failure.hidden = state !== "error";
    this.retry.hidden = state !== "error";
    this.download.hidden = state !== "error";
    this.video.dispatchEvent(new Event("videoloadstate"));
  }

  clear() {
    this.task?.abort();
    this.task = null;
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    if (this.objectURL) URL.revokeObjectURL(this.objectURL);
    this.objectURL = null;
    this.setState("idle");
  }

  async load(source, { time = 0 } = {}) {
    this.clear();
    this.source = source;
    this.position = time;
    this.download.href = source;
    const task = new AbortController();
    this.task = task;
    this.progress.removeAttribute("value");
    this.setState("loading");
    try {
      const blob = await new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open("GET", source);
        request.responseType = "blob";
        request.timeout = 120000;
        const abort = () => request.abort();
        task.signal.addEventListener("abort", abort, { once: true });
        request.addEventListener("loadend", () => task.signal.removeEventListener("abort", abort), { once: true });
        request.addEventListener("progress", (event) => {
          if (event.lengthComputable && !task.signal.aborted) {
            this.progress.value = Math.min(99, event.loaded / event.total * 100);
          }
        });
        request.addEventListener("load", () => {
          if (request.status !== 200) reject(new Error(`El servidor devolvió HTTP ${request.status}.`));
          else if (!request.response?.size) reject(new Error("El archivo está vacío."));
          else resolve(request.response);
        });
        request.addEventListener("error", () => reject(new Error("No hay conexión con el vídeo.")));
        request.addEventListener("timeout", () => reject(new Error("La descarga ha superado los dos minutos.")));
        request.addEventListener("abort", () => reject(new DOMException("Carga cancelada.", "AbortError")));
        request.send();
      });
      if (task.signal.aborted) return false;
      this.progress.value = 100;
      this.status.textContent = "Vídeo descargado. Preparando reproducción…";
      // Some hosts label MP4 assets as octet-stream.
      this.objectURL = URL.createObjectURL(new Blob([blob], { type: "video/mp4" }));
      await new Promise((resolve, reject) => {
        const video = this.video;
        let positioned = false;
        const finish = (failure) => {
          clearTimeout(timer);
          for (const name of ["loadedmetadata", "loadeddata", "canplay", "seeked"]) video.removeEventListener(name, prepare);
          video.removeEventListener("error", failed);
          task.signal.removeEventListener("abort", aborted);
          if (failure) reject(failure);
          else resolve();
        };
        const prepare = () => {
          if (video.readyState >= 1 && this.position > 0 && !positioned) {
            const position = Math.min(this.position, video.duration);
            positioned = true;
            video.currentTime = position;
          }
          if (video.readyState >= 3 && !video.seeking) finish();
        };
        const failed = () => finish(new Error("El navegador no puede reproducir este archivo."));
        const aborted = () => finish(new DOMException("Carga cancelada.", "AbortError"));
        const timer = setTimeout(() => finish(new Error("El navegador no ha podido preparar el vídeo a tiempo.")), 30000);
        for (const name of ["loadedmetadata", "loadeddata", "canplay", "seeked"]) video.addEventListener(name, prepare);
        video.addEventListener("error", failed, { once: true });
        task.signal.addEventListener("abort", aborted, { once: true });
        video.preload = "auto";
        video.src = this.objectURL;
        video.load();
      });
      if (task.signal.aborted) return false;
      this.setState("ready");
      return true;
    } catch (failure) {
      if (task.signal.aborted) return false;
      console.error("Carga de vídeo:", failure);
      this.video.removeAttribute("src");
      this.video.load();
      if (this.objectURL) URL.revokeObjectURL(this.objectURL);
      this.objectURL = null;
      this.setState("error", `No se ha podido cargar el vídeo. ${failure.message}`);
      return false;
    }
  }
};

function setupVideoPair(controls) {
  const [original, analysis] = controls.dataset.videoPair.split(" ").map((id) => document.getElementById(id));
  const videos = [original, analysis];
  const buttons = [...controls.querySelectorAll("[data-video-action]")];
  const error = controls.querySelector("[data-pair-error]");
  const ready = () => videos.every((video) => video.closest(".video-loader")?.dataset.state === "ready");
  let linked = false, pending = false, attempt = 0;
  const pause = () => {
    attempt++;
    linked = false;
    pending = false;
    videos.forEach((video) => video.pause());
  };
  const refresh = () => {
    const available = ready();
    if (!available) pause();
    for (const button of buttons) {
      button.disabled = !available || (pending && button.dataset.videoAction === "play");
    }
  };
  const align = () => {
    if (ready() && Math.abs(analysis.currentTime - original.currentTime) > 0.15) {
      analysis.currentTime = original.currentTime;
    }
  };
  for (const button of buttons) button.addEventListener("click", async () => {
    if (button.dataset.videoAction !== "play") {
      pause();
      if (button.dataset.videoAction === "restart") {
        videos.forEach((video) => { video.currentTime = 0; });
        error.hidden = true;
      }
      refresh();
      return;
    }
    error.hidden = true;
    if (videos.some((video) => video.ended)) original.currentTime = 0;
    analysis.currentTime = original.currentTime;
    analysis.playbackRate = original.playbackRate;
    const current = ++attempt;
    linked = true;
    pending = true;
    refresh();
    try {
      await Promise.all(videos.map((video) => video.play()));
    } catch (failure) {
      if (current !== attempt) return;
      pause();
      console.error("Reproducción conjunta:", failure);
      error.textContent = `No se han podido reproducir ambos vídeos. Pulsa reproducir de nuevo: ${failure.message}`;
      error.hidden = false;
      refresh();
    } finally {
      if (current === attempt) {
        pending = false;
        refresh();
      }
    }
  });
  original.addEventListener("timeupdate", () => { if (linked) align(); });
  original.addEventListener("seeked", align);
  original.addEventListener("ratechange", () => {
    if (linked) analysis.playbackRate = original.playbackRate;
  });
  for (const video of videos) {
    video.addEventListener("videoloadstate", refresh);
    video.addEventListener("pause", () => {
      if (linked && video.paused) { pause(); refresh(); }
    });
    video.addEventListener("ended", () => { pause(); refresh(); });
  }
  refresh();
}

document.addEventListener("DOMContentLoaded", () => {
  for (const video of document.querySelectorAll("video[data-video-src]")) {
    const loader = new window.TennisVideoLoader(video);
    void loader.load(video.dataset.videoSrc);
  }
  for (const controls of document.querySelectorAll("[data-video-pair]")) setupVideoPair(controls);
});
