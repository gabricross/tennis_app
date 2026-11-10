"use strict";

(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const resultLoader = new window.TennisVideoLoader($("#result-video"));
  const state = {
    config: null, clips: [], jobs: [], clip: null, job: null, result: null,
    screen: "studio", stage: "upload", points: [], confirmedFrame: null,
    previewSerial: 0, dragging: -1, xhr: null, mapMode: "positions",
    resultView: "analysis", polling: false, densityCache: new Map(),
    expandedClips: new Set(), calibration: null, calibrationKey: "", calibrationError: "",
  };
  const names = ["Arriba izquierda", "Arriba derecha", "Abajo derecha", "Abajo izquierda"];
  const statusNames = {
    queued: "En cola", running: "Analizando", cancelling: "Cancelando", cancelled: "Cancelado",
    failed: "No completado", interrupted: "Interrumpido", completed: "Terminado",
    completed_with_warnings: "Terminado con avisos",
  };
  const active = (job) => ["queued", "running", "cancelling"].includes(job?.status);
  const finished = (job) => ["completed", "completed_with_warnings"].includes(job?.status);
  const seconds = (value) => {
    const minutes = Math.floor(value / 60);
    return `${String(minutes).padStart(2, "0")}:${(value % 60).toFixed(1).padStart(4, "0")}`;
  };
  const percent = (value) => `${Math.round(value * 100)}%`;
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const error = (message) => {
    $("#global-message").textContent = message;
    $("#global-message").hidden = false;
    console.error(message);
  };
  const clearError = () => { $("#global-message").hidden = true; };
  let toastTimer;
  const toast = (message) => {
    $("#toast").textContent = message;
    $("#toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 5000);
  };
  const run = (action) => {
    clearError();
    Promise.resolve().then(action).catch((failure) => error(failure.message));
  };
  async function api(path, options = {}) {
    const headers = new Headers(options.headers);
    if (options.method && options.method !== "GET") headers.set("X-Tennis-Token", state.config.token);
    if (options.body && typeof options.body === "string") headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...options, headers });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `El servidor devolvió un error ${response.status}.`);
    return body;
  }
  const asset = (name) => `/api/jobs/${state.job.id}/files/${name}`;

  function screen(name) {
    state.screen = name;
    $$(".screen").forEach((node) => { node.hidden = node.id !== `screen-${name}`; });
    $$("nav [data-screen]").forEach((button) => {
      if (button.dataset.screen === name) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    $("#screen-title").textContent = { studio: "Estudio de análisis", library: "Biblioteca local", about: "Motor y privacidad" }[name];
    if (name === "library") renderLibrary();
  }
  function stage(name) {
    state.stage = name;
    $$(".stage").forEach((node) => { node.hidden = node.id !== `${name}-stage`; });
    $(".studio-heading").hidden = name !== "upload";
    const step = { upload: 1, configure: 2, progress: 3, results: 4 }[name];
    $$(".workflow li").forEach((node) => {
      node.classList.toggle("active", Number(node.dataset.step) === step);
      node.classList.toggle("done", Number(node.dataset.step) < step);
    });
  }
  function newAnalysis() {
    clearError();
    resultLoader.clear();
    state.clip = null;
    state.points = [];
    state.confirmedFrame = null;
    state.result = null;
    stage("upload");
    screen("studio");
  }
  async function refreshLibrary() {
    const data = await api("/api/library");
    state.clips = data.clips;
    state.jobs = data.jobs;
    $("#library-count").textContent = String(data.clips.length);
    if (state.screen === "library") renderLibrary();
  }
  function renderLibrary() {
    $("#library-list").replaceChildren();
    $("#library-empty").hidden = state.clips.length > 0;
    for (const clip of state.clips) {
      const card = element("article", undefined, "library-card");
      const image = element("img");
      image.src = `/api/clips/${clip.id}/poster`;
      image.alt = `Vista previa de ${clip.name}`;
      image.loading = "lazy";
      const info = element("div");
      info.append(element("h3", clip.name), element("p",
        `${seconds(clip.metadata.duration)} · ${clip.metadata.width} × ${clip.metadata.height} · ${(clip.metadata.bytes / 1048576).toFixed(1)} MB`));
      const jobs = state.jobs.filter((job) => job.clip_id === clip.id);
      const actions = element("div", undefined, "library-actions");
      const configure = element("button", jobs.length ? "Analizar de nuevo" : "Preparar clip", "button secondary");
      configure.addEventListener("click", () => run(() => loadClip(clip, jobs[0]?.config)));
      actions.append(configure);
      const visibleJobs = state.expandedClips.has(clip.id) ? jobs : jobs.slice(0, 4);
      for (const job of visibleJobs) {
        const button = element("button", `${statusNames[job.status]} · ${job.config.mode === "complete" ? "Pelota" : "Jugadores"}`, "button secondary");
        button.addEventListener("click", () => run(async () => {
          state.clip = clip;
          state.job = await api(`/api/jobs/${job.id}`);
          screen("studio");
          if (finished(state.job)) await showResult(state.job);
          else { stage("progress"); showProgress(state.job); }
        }));
        actions.append(button);
      }
      if (jobs.length > 4) {
        const expand = element("button", state.expandedClips.has(clip.id)
          ? "Mostrar solo los recientes" : `Ver los ${jobs.length} análisis`, "text-button");
        expand.addEventListener("click", () => {
          if (state.expandedClips.has(clip.id)) state.expandedClips.delete(clip.id);
          else state.expandedClips.add(clip.id);
          renderLibrary();
        });
        actions.append(expand);
      }
      info.append(actions);
      const remove = element("button", "Borrar clip y resultados", "delete-clip");
      remove.disabled = jobs.some(active);
      remove.title = remove.disabled ? "Cancela sus análisis antes de borrarlo." : "";
      remove.addEventListener("click", () => run(async () => {
        if (!confirm(`¿Borrar definitivamente «${clip.name}» y todos sus resultados locales?`)) return;
        await api(`/api/clips/${clip.id}`, { method: "DELETE" });
        if (state.clip?.id === clip.id) {
          state.clip = null;
          state.result = null;
          resultLoader.clear();
          stage("upload");
        }
        await refreshLibrary();
        toast("Clip y resultados eliminados de este equipo.");
      }));
      card.append(image, info, remove);
      $("#library-list").append(card);
    }
  }

  async function upload(file) {
    if (!state.config) throw new Error("El motor todavía no ha conectado.");
    if (state.xhr) throw new Error("Espera a que termine la carga actual.");
    if (file.size > state.config.max_upload_mb * 1048576) throw new Error("El vídeo supera los 300 MB.");
    if (!/\.(mp4|mov|webm|mkv|avi)$/i.test(file.name)) throw new Error("Selecciona un archivo MP4, MOV, WebM, MKV o AVI.");
    $("#upload-progress").hidden = false;
    $("#choose-file").disabled = true;
    $("#upload-bar").value = 0;
    $("#upload-status").textContent = "Copiando vídeo al almacenamiento privado local…";
    try {
      const clip = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        state.xhr = xhr;
        xhr.open("POST", "/api/clips");
        xhr.setRequestHeader("X-Tennis-Token", state.config.token);
        xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
        xhr.setRequestHeader("Content-Type", "application/octet-stream");
        xhr.upload.addEventListener("progress", (event) => {
          if (event.lengthComputable) {
            $("#upload-bar").value = event.loaded / event.total * 100;
            $("#upload-status").textContent = event.loaded === event.total
              ? "Vídeo recibido. Verificando formato y creando vista previa…"
              : `Carga local: ${Math.round(event.loaded / event.total * 100)}%`;
          }
        });
        xhr.addEventListener("load", () => {
          let body;
          try { body = JSON.parse(xhr.responseText); }
          catch (failure) { reject(new Error(`El servidor no devolvió una respuesta válida (${xhr.status}).`)); return; }
          if (xhr.status >= 200 && xhr.status < 300) resolve(body);
          else reject(new Error(body.detail || "No se ha podido cargar el vídeo."));
        });
        xhr.addEventListener("error", () => reject(new Error("El servidor local no responde. Comprueba la ventana de Tennis Lab.")));
        xhr.addEventListener("abort", () => reject(new DOMException("Carga cancelada; no se conserva el archivo parcial.", "AbortError")));
        xhr.send(file);
      });
      await refreshLibrary();
      await loadClip(clip);
    } catch (failure) {
      if (failure.name === "AbortError") toast(failure.message);
      else throw failure;
    } finally {
      state.xhr = null;
      $("#choose-file").disabled = false;
      $("#upload-progress").hidden = true;
      $("#file-input").value = "";
    }
  }

  async function loadClip(clip, previous) {
    state.clip = clip;
    state.points = [];
    $("#clip-start").value = previous?.start ?? 0;
    $("#clip-duration").value = previous?.duration ?? Math.min(30, clip.metadata.duration);
    $("#clip-description").textContent = `${clip.name} · ${seconds(clip.metadata.duration)} · ${clip.metadata.width} × ${clip.metadata.height}`;
    $("#input-notice").hidden = !clip.metadata.input_notice;
    $("#input-notice").textContent = clip.metadata.input_notice || "";
    $("#confirm-court").checked = false;
    $("#calibration-reference").value = previous?.calibration_reference ?? "doubles";
    $("#ball-backend").value = previous?.ball_backend ?? (state.config.models.ball_yolo ? "yolo_multiscale" : "tracknet");
    $("#players-mode").checked = previous?.mode === "players" || (previous?.mode !== "complete" && $("#complete-mode").disabled);
    $("#complete-mode").checked = !$("#players-mode").checked;
    referenceGuide();
    screen("studio");
    stage("configure");
    await loadPreview();
    if (previous?.corners) {
      state.points = previous.corners.map((point) => [...point]);
      drawCalibration();
    }
  }
  function referenceGuide() {
    const reference = $("#calibration-reference").value;
    if (reference === "doubles") {
      $("#calibration-image").alt = "Marca las cuatro esquinas exteriores de dobles visibles.";
      $("#reference-title").textContent = "Las líneas exteriores, no las de individuales.";
      $("#reference-help").textContent = "Marca las cuatro esquinas de dobles que realmente se ven.";
    } else if (reference === "service_boxes") {
      $("#calibration-image").alt = "Marca los cuatro extremos del rectángulo de saque completo, a ambos lados de la red.";
      $("#reference-title").textContent = "Todo el rectángulo de saque, NO los fondos.";
      $("#reference-help").textContent = "Marca las dos intersecciones de cada línea de saque con los laterales de individuales: "
        + "dos puntos lejanos y dos cercanos, incluyendo ambas columnas. NO marques la cinta de la red. "
        + "Es una referencia más amplia que una sola columna, pero los fondos siguen siendo una extrapolación.";
    } else {
      const side = reference === "service_left" ? "izquierda" : "derecha";
      $("#calibration-image").alt = `Marca las cuatro intersecciones visibles de la columna ${side} de cuadros de saque, no la red.`;
      $("#reference-title").textContent = `Solo la columna ${side} de saque, NO toda la pista.`;
      $("#reference-help").textContent = "El rectángulo verde NO es el contorno del campo. Marca las intersecciones de la línea de saque lejana y la cercana "
        + `con la línea central y el lateral de individuales de la ${side}. `
        + "Son los cuatro extremos de un rectángulo que cruza la red; NO marques la cinta de la red. "
        + "El resto de la pista se extrapola: revisa también las líneas turquesas. Si no encajan, cambia los puntos o la toma.";
    }
    const court = state.config.court;
    const canvas = $("#reference-diagram"), context = canvas.getContext("2d");
    const xy = ([x, y]) => [30 + x / court.width_m * 200, 30 + y / court.length_m * 480];
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#70dfd2"; context.lineWidth = 2;
    court.lines.forEach(([a, b]) => {
      context.beginPath(); context.moveTo(...xy(a)); context.lineTo(...xy(b)); context.stroke();
    });
    context.setLineDash([6, 6]); context.beginPath();
    context.moveTo(...xy([0, court.length_m / 2]));
    context.lineTo(...xy([court.width_m, court.length_m / 2])); context.stroke(); context.setLineDash([]);
    context.font = "14px Segoe UI"; context.textAlign = "center"; context.fillStyle = "#ecf3ec";
    context.fillText("RED", 130, 260);
    const points = court.references[reference].map(xy);
    context.beginPath(); points.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
    context.closePath(); context.fillStyle = "#d9fa7326"; context.fill();
    context.strokeStyle = "#d9fa73"; context.stroke();
    points.forEach(([x, y], i) => {
      context.beginPath(); context.arc(x, y, 13, 0, Math.PI * 2); context.fillStyle = "#d9fa73"; context.fill();
      context.fillStyle = "#15231c"; context.font = "bold 17px Segoe UI"; context.textBaseline = "middle";
      context.fillText(String(i + 1), x, y);
    });
  }
  function range() {
    const start = Number($("#clip-start").value);
    const duration = Number($("#clip-duration").value);
    if (!Number.isFinite(start) || !Number.isFinite(duration) || start < 0 || duration < 1 || duration > state.config.max_analysis_seconds) {
      throw new Error("Elige un inicio válido y una duración entre 1 y 120 segundos.");
    }
    if (start + duration > state.clip.metadata.duration + 0.02) throw new Error("El fragmento termina después del vídeo.");
    return { start, duration };
  }
  async function loadPreview() {
    const selected = range();
    const serial = ++state.previewSerial;
    state.confirmedFrame = null;
    state.points = [];
    $("#confirm-court").checked = false;
    $("#load-frame").disabled = true;
    $("#corner-instruction").textContent = "Leyendo el fotograma…";
    $("#range-description").textContent = `${seconds(selected.start)} → ${seconds(selected.start + selected.duration)} del vídeo original.`;
    const image = $("#calibration-image");
    try {
      await new Promise((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error("No se ha podido leer el fotograma seleccionado."));
        image.src = `/api/clips/${state.clip.id}/frame?time=${selected.start}`;
      });
      if (serial !== state.previewSerial) return;
      $("#calibration-canvas").width = image.naturalWidth;
      $("#calibration-canvas").height = image.naturalHeight;
      state.confirmedFrame = selected.start;
      drawCalibration();
    } finally {
      $("#load-frame").disabled = false;
      updateReadiness();
    }
  }
  function updateCornerFields() {
    const container = $("#corner-fields");
    container.replaceChildren();
    state.points.forEach((point, index) => {
      const row = element("div", undefined, "corner-row");
      row.append(element("span", String(index + 1)));
      ["X", "Y"].forEach((axis, component) => {
        const input = element("input");
        input.type = "number"; input.min = "0"; input.max = "100"; input.step = "0.1";
        input.value = (point[component] * 100).toFixed(1);
        input.setAttribute("aria-label", `${names[index]}, ${axis} en porcentaje`);
        input.addEventListener("change", () => {
          const value = Number(input.value);
          if (!Number.isFinite(value) || value < 0 || value > 100) {
            error("Las coordenadas deben estar entre 0 y 100%."); return;
          }
          state.points[index][component] = value / 100;
          $("#confirm-court").checked = false;
          drawCalibration(false);
        });
        row.append(input);
      });
      container.append(row);
    });
  }
  let calibrationTimer;
  function queueCalibrationPreview() {
    const body = { time: state.confirmedFrame, corners: state.points, calibration_reference: $("#calibration-reference").value };
    const key = JSON.stringify([state.clip?.id, body]);
    if (key === state.calibrationKey) return;
    state.calibrationKey = key; state.calibration = null; state.calibrationError = "";
    $("#confirm-court").checked = false;
    clearTimeout(calibrationTimer);
    $("#calibration-quality").className = "notice warning";
    $("#calibration-quality").textContent = "Marca los cuatro puntos para previsualizar todas las líneas.";
    if (state.points.length !== 4 || state.confirmedFrame === null) return;
    const payload = JSON.stringify(body), identity = state.clip.id;
    $("#calibration-quality").textContent = "Comprobando la proyección de toda la pista…";
    calibrationTimer = setTimeout(async () => {
      try {
        const preview = await api(`/api/clips/${identity}/calibration`, { method: "POST", body: payload });
        if (state.calibrationKey !== key) return;
        state.calibration = preview;
        $("#calibration-quality").textContent = !preview.usable
          ? "Las líneas proyectadas tienen poco apoyo en las líneas blancas. Revisa los puntos o usa una toma más clara; no se puede confirmar así."
          : (preview.partial ? "Calibración PARCIAL: el rectángulo verde solo es la referencia; las líneas turquesas discontinuas extrapolan el resto de la pista. "
            : "Comprueba que TODAS las líneas turquesas coinciden con la pista, no solo las cuatro esquinas. ")
            + `Apoyo orientativo en líneas: ${percent(preview.support)}; NO mide precisión.`;
        $("#calibration-quality").className = preview.partial || !preview.usable ? "notice warning" : "notice";
      } catch (failure) {
        if (state.calibrationKey !== key) return;
        state.calibrationError = failure.message;
        $("#calibration-quality").textContent = failure.message;
        $("#calibration-quality").className = "notice error";
      }
      drawCalibration(false);
    }, 250);
  }
  function drawCalibration(rebuildFields = true) {
    queueCalibrationPreview();
    const canvas = $("#calibration-canvas");
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (state.calibration) {
      context.strokeStyle = "#70dfd2"; context.lineWidth = Math.max(2, canvas.width / 400);
      context.setLineDash(state.calibration.partial ? [12, 8] : []);
      state.calibration.lines.forEach(([a, b]) => {
        context.beginPath(); context.moveTo(a[0] * canvas.width, a[1] * canvas.height);
        context.lineTo(b[0] * canvas.width, b[1] * canvas.height); context.stroke();
      });
      context.setLineDash([]);
    }
    if (state.points.length > 1) {
      context.beginPath();
      state.points.forEach((point, index) => {
        const x = point[0] * canvas.width, y = point[1] * canvas.height;
        if (index) context.lineTo(x, y); else context.moveTo(x, y);
      });
      if (state.points.length === 4) context.closePath();
      context.strokeStyle = "#d9fa73"; context.lineWidth = Math.max(2, canvas.width / 450);
      context.stroke();
      if (state.points.length === 4) { context.fillStyle = "#d9fa7314"; context.fill(); }
    }
    state.points.forEach((point, index) => {
      const x = point[0] * canvas.width, y = point[1] * canvas.height;
      const radius = Math.max(11, canvas.width / 75);
      context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2);
      context.fillStyle = "#d9fa73"; context.fill();
      context.fillStyle = "#15231c"; context.font = `bold ${radius * 1.15}px Segoe UI`;
      context.textAlign = "center"; context.textBaseline = "middle";
      context.fillText(String(index + 1), x, y);
    });
    $("#corner-instruction").textContent = state.points.length < 4
      ? `Marca el punto ${state.points.length + 1}: ${names[state.points.length]}.`
      : "Revisa los cuatro puntos. Arrástralos para ajustar o edita sus coordenadas.";
    $$("#corner-list li").forEach((item, index) => item.classList.toggle("selected", index === state.points.length));
    if (rebuildFields) updateCornerFields();
    updateReadiness();
  }
  function updateReadiness() {
    const complete = $("#complete-mode").checked;
    $("#ball-options").hidden = !complete;
    $("#analysis-plan").textContent = complete
      ? "Se analizarán PISTA + JUGADORES + PELOTA juntos en un único vídeo."
      : "Se analizarán PISTA + JUGADORES. La pelota NO está incluida.";
    $("#start-analysis").textContent = complete ? "Analizar pista + jugadores + pelota →" : "Analizar pista + jugadores →";
    $("#confirm-court").disabled = !state.calibration?.usable;
    let message = "Marca las cuatro esquinas y revisa la pista.";
    let ready = false;
    try {
      const selected = range();
      if (state.confirmedFrame !== selected.start) message = "Actualiza el fotograma: has cambiado el inicio del fragmento.";
      else if (state.points.length === 4 && !state.calibration?.usable) {
        message = state.calibrationError || (state.calibration ? "Corrige la proyección antes de confirmar." : "Esperando la comprobación de la pista.");
      } else if (state.points.length === 4 && $("#confirm-court").checked) {
        ready = true;
        message = `Se analizarán ${selected.duration.toFixed(1)} s. El archivo original no se modifica.`;
      } else if (state.points.length === 4) message = "Confirma que las esquinas corresponden a esta toma.";
    } catch (failure) { message = failure.message; }
    if (state.config && !state.config.models.people) {
      ready = false;
      message = "Falta el modelo de jugadores. Ejecuta Preparar_Tennis_Lab.ps1.";
    }
    if (state.config && $("#complete-mode").checked) {
      const keys = {tracknet: ["ball"], yolo_wasb: ["ball_yolo", "ball_wasb"]}[$("#ball-backend").value] ?? ["ball_yolo"];
      if (!keys.every((key) => state.config.models[key])) {
        ready = false;
        message = "No está instalado el motor de pelota seleccionado. Elige otro o instala sus pesos.";
      }
    }
    $("#analysis-readiness").textContent = message;
    $("#start-analysis").disabled = !ready;
    $("#calibration-status").textContent = ready ? "Confirmada" : "Revisar";
  }

  async function startJob() {
    const selected = range();
    if (state.points.length !== 4 || !$("#confirm-court").checked || state.confirmedFrame !== selected.start) {
      throw new Error("Revisa y confirma la pista antes de continuar.");
    }
    const mode = $("input[name=analysis-mode]:checked").value;
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        clip_id: state.clip.id, ...selected, mode, corners: state.points,
        calibration_reference: $("#calibration-reference").value,
        ball_backend: $("#ball-backend").value,
      }),
    });
    state.job = job;
    stage("progress");
    showProgress(job);
    await refreshLibrary();
  }
  function showProgress(job) {
    $("#job-message").textContent = job.error || job.message;
    $("#job-percent").replaceChildren(document.createTextNode(String(Math.round(job.progress))), element("span", "%"));
    $("#job-progress").value = job.progress;
    $("#progress-title").textContent = job.status === "failed" ? "Este clip necesita revisión."
      : job.status === "cancelled" ? "Análisis cancelado." : "Cada fotograma cuenta.";
    $("#cancel-job").disabled = job.status === "cancelling";
    $("#cancel-job").textContent = active(job) ? "Cancelar análisis" : "Volver a configurar";
    $("#job-elapsed").textContent = `${Math.max(0, Math.floor((Date.now() - Date.parse(job.created)) / 1000))} s desde la entrada en cola`;
  }
  async function poll() {
    if (state.polling || !state.job || !active(state.job)) return;
    state.polling = true;
    try {
      const job = await api(`/api/jobs/${state.job.id}`);
      state.job = job;
      if (state.stage === "progress") showProgress(job);
      if (!active(job)) {
        await refreshLibrary();
        if (finished(job)) {
          if (state.screen === "studio" && state.stage === "progress") await showResult(job);
          else toast("Un análisis ha terminado. Ya está en tu biblioteca.");
        } else if (job.status === "failed") error(job.error || job.message);
        else toast(job.message);
      }
    } catch (failure) { error(`No se ha podido consultar el progreso: ${failure.message}`); }
    finally { state.polling = false; }
  }

  function drawZones(side, values) {
    const container = $(`#${side}-zones`);
    container.replaceChildren();
    for (const [key, label] of [["baseline", "Fondo"], ["middle", "Media"], ["net", "Red"]]) {
      const row = element("div", undefined, "zone-row");
      const rail = element("div", undefined, "zone-rail");
      const fill = element("div", undefined, "zone-fill");
      fill.style.width = percent(values[key]);
      rail.append(fill);
      row.append(element("span", label), rail, element("span", percent(values[key])));
      container.append(row);
    }
  }
  async function showResult(job) {
    state.job = job;
    state.result = await api(asset("analysis.json"));
    state.densityCache.clear();
    state.resultView = "analysis";
    state.mapMode = "positions";
    $("#player-filter").value = "both";
    screen("studio");
    stage("results");
    const { summary } = state.result;
    const clip = state.clips.find((item) => item.id === job.clip_id) || state.clip;
    const warningsText = [...summary.warnings];
    if (clip?.metadata.input_notice && !warningsText.includes(clip.metadata.input_notice)) {
      warningsText.unshift(clip.metadata.input_notice);
    }
    $("#result-subtitle").textContent = `${clip?.name || "Tu clip"} · ${summary.duration.toFixed(1)} s · ${summary.mode === "complete" ? "Jugadores y pelota experimental" : "Jugadores y pista"} · Procesamiento: ${Math.round(summary.total_seconds_including_render)} s`;
    const warnings = $("#result-warnings");
    warnings.replaceChildren();
    warnings.hidden = !warningsText.length;
    if (warningsText.length) {
      warnings.append(element("strong", "Qué debes tener en cuenta"));
      const list = element("ul");
      warningsText.forEach((warning) => list.append(element("li", warning)));
      warnings.append(list);
    }
    for (const side of ["north", "south"]) {
      $(`#${side}-coverage`).textContent = percent(summary.players[side].coverage);
      drawZones(side, summary.players[side].zones);
      const movement = summary.players[side].movement;
      const number = (value) => typeof value === "number" ? value.toFixed(1) : "—";
      $(`#${side}-motion`).textContent = movement
        ? `Distancia observada: ${number(movement.distance_m)} m · Media: ${number(movement.mean_speed_kmh)} km/h · Pico de ventana: ${number(movement.peak_speed_kmh)} km/h · ${number(movement.observed_seconds)} s evaluables. Estimaciones, no medidas certificadas.`
        : "Análisis antiguo: vuelve a analizar para estimar velocidad y distancia con la geometría corregida.";
    }
    $("#result-metrics-note").textContent = summary.metrics_note;
    $("#result-seek").max = String(summary.duration);
    const video = $("#result-video");
    video.pause();
    video.poster = asset("poster.jpg");
    void resultLoader.load(asset("analysis.mp4"));
    $$("[data-result-view]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.resultView === "analysis")));
    $$("[data-map-mode]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mapMode === "positions")));
    $("#result-view-label").textContent = "Vídeo analizado";
    $("#export-links").replaceChildren();
    const comparison = element("a", "Comparar ambos ↗");
    comparison.href = `/api/jobs/${job.id}/comparison`;
    comparison.target = "_blank";
    comparison.rel = "noopener";
    $("#export-links").append(comparison);
    for (const [file, label] of [["analysis.mp4", "MP4"], ["observations.csv", "CSV"], ["analysis.json", "JSON"], ["report.zip", "Paquete ZIP"]]) {
      const link = element("a", `${label} ↓`);
      link.href = asset(file);
      link.download = `tennis-lab-${job.id.slice(0, 8)}-${file}`;
      $("#export-links").append(link);
    }
    if (summary.mode === "complete") {
      const link = element("a", "Pelota CSV ↓");
      link.href = asset("ball.csv");
      link.download = `tennis-lab-${job.id.slice(0, 8)}-ball.csv`;
      $("#export-links").append(link);
    }
    updateResult();
  }
  const worldPoint = ([x, y]) => [98 + x * 27.7, 166 + y * 27.7];
  function courtDrawing(context) {
    context.fillStyle = "#102126"; context.fillRect(0, 0, 500, 1000);
    for (let stripe = 0; stripe < 10; stripe++) {
      context.fillStyle = stripe % 2 ? "#214940" : "#254f43";
      const [x, y] = worldPoint([0, stripe * 2.377]);
      context.fillRect(x, y, 10.97 * 27.7, 2.377 * 27.7);
    }
    context.strokeStyle = "#a7c4aa"; context.lineWidth = 2;
    const segment = (a, b) => { context.beginPath(); context.moveTo(...worldPoint(a)); context.lineTo(...worldPoint(b)); context.stroke(); };
    for (const x of [0, 1.37, 9.60, 10.97]) segment([x, 0], [x, 23.77]);
    for (const y of [0, 23.77]) segment([0, y], [10.97, y]);
    const service = state.result?.version >= 3 ? [23.77 / 2 - 6.4, 23.77 / 2 + 6.4] : [6.4, 17.37];
    for (const y of service) segment([1.37, y], [9.60, y]);
    segment([10.97 / 2, service[0]], [10.97 / 2, service[1]]);
    context.strokeStyle = "#d9fa73"; segment([0, 11.885], [10.97, 11.885]);
  }
  function resultSample(time) {
    let result = state.result.samples[0];
    for (const sample of state.result.samples) { if (sample.time > time) break; result = sample; }
    return result;
  }
  function drawMap(sample) {
    const canvas = $("#result-map"), context = canvas.getContext("2d");
    const filter = $("#player-filter").value;
    courtDrawing(context);
    if (state.mapMode === "density") {
      if (!state.densityCache.has(filter)) {
        const cache = document.createElement("canvas"); cache.width = 500; cache.height = 1000;
        const drawing = cache.getContext("2d");
        for (const entry of state.result.samples) {
          for (const player of entry.players) {
            if (filter !== "both" && player.side !== filter) continue;
            const [x, y] = worldPoint(player.pitch);
            const gradient = drawing.createRadialGradient(x, y, 1, x, y, 24);
            gradient.addColorStop(0, player.side === "north" ? "#70dfd225" : "#ffa16f25");
            gradient.addColorStop(1, "#00000000");
            drawing.fillStyle = gradient; drawing.fillRect(x - 24, y - 24, 48, 48);
          }
        }
        state.densityCache.set(filter, cache);
      }
      context.drawImage(state.densityCache.get(filter), 0, 0);
      $("#map-caption").textContent = "Acumulado del clip completo. Intensidad = número de observaciones, no esfuerzo ni tiempo sin huecos.";
    } else {
      for (const player of sample.players) {
        if (filter !== "both" && player.side !== filter) continue;
        const [x, y] = worldPoint(player.pitch);
        context.beginPath(); context.arc(x, y, 10, 0, Math.PI * 2);
        context.fillStyle = player.side === "north" ? "#70dfd2" : "#ffa16f";
        context.fill(); context.strokeStyle = "#102126"; context.lineWidth = 3; context.stroke();
      }
      $("#map-caption").textContent = sample.camera_valid
        ? "Apoyos estimados en este instante. La pelota no se proyecta al suelo."
        : "Registro perdido en este instante. No se muestran posiciones inventadas.";
    }
  }
  function drawTimeline(time) {
    const canvas = $("#coverage-timeline"), context = canvas.getContext("2d");
    const samples = state.result.samples, duration = state.result.summary.duration;
    context.fillStyle = "#132329"; context.fillRect(0, 0, 1200, 114);
    for (const [row, key, label, color] of [[0, "north", "ARRIBA", "#70dfd2"], [1, "south", "ABAJO", "#ffa16f"], [2, "ball", "PELOTA", "#d9fa73"]]) {
      context.font = "14px Segoe UI"; context.fillStyle = "#a2b7ba"; context.fillText(label, 0, row * 35 + 20);
      context.fillStyle = "#2c4249"; context.fillRect(80, row * 35 + 4, 1120, 23);
      const series = key === "ball" ? (state.result.ball_samples ?? samples) : samples;
      for (let index = 0; index < series.length; index++) {
        const entry = series[index];
        const observed = key === "ball" ? entry.ball !== null : entry.players.some((player) => player.side === key);
        if (!observed) continue;
        const end = series[index + 1]?.time ?? duration;
        context.fillStyle = color;
        context.fillRect(80 + entry.time / duration * 1120, row * 35 + 4, Math.max(1, (end - entry.time) / duration * 1120), 23);
      }
    }
    const x = 80 + time / duration * 1120;
    context.strokeStyle = "#ffffff"; context.lineWidth = 2;
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, 104); context.stroke();
  }
  function updateResult() {
    if (!state.result) return;
    const time = $("#result-video").currentTime || 0;
    $("#result-seek").value = String(time);
    $("#result-time").textContent = `${seconds(time)} del clip · ${seconds(state.job.config.start + time)} del original`;
    $("#map-time").textContent = seconds(time);
    const sample = resultSample(time);
    for (const side of ["north", "south"]) {
      const player = sample.camera_valid && time - sample.time <= 0.25
        ? sample.players.find((entry) => entry.side === side) : null;
      const speed = player?.motion?.speed_kmh;
      $(`#${side}-current-speed`).textContent = typeof speed === "number"
        ? `En este instante: ${speed.toFixed(1)} km/h estimados.`
        : "En este instante: velocidad no disponible.";
    }
    drawMap(sample);
    drawTimeline(time);
  }
  async function switchResultView(view) {
    if (!state.result || state.resultView === view) return;
    const video = $("#result-video"), time = resultLoader.time, playing = !video.paused;
    video.pause();
    state.resultView = view;
    $$("[data-result-view]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.resultView === view)));
    $("#result-view-label").textContent = view === "analysis" ? "Vídeo analizado" : "Fragmento original normalizado";
    video.removeAttribute("poster");
    const ready = await resultLoader.load(asset(view === "analysis" ? "analysis.mp4" : "original.mp4"), { time });
    if (ready && playing) video.play().catch((failure) => error(`Pulsa reproducir para continuar: ${failure.message}`));
  }

  $$("[data-screen]").forEach((button) => button.addEventListener("click", () => {
    if (button.id === "library-new") newAnalysis();
    else screen(button.dataset.screen);
  }));
  $("#choose-file").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (event) => { if (event.target.files[0]) run(() => upload(event.target.files[0])); });
  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => {
    if (event.dataTransfer.files.length !== 1) { error("Carga un único vídeo cada vez."); return; }
    run(() => upload(event.dataTransfer.files[0]));
  });
  $("#cancel-upload").addEventListener("click", () => state.xhr?.abort());
  $("#use-previous").addEventListener("click", () => run(async () => {
    $("#use-previous").disabled = true;
    try { const clip = await api("/api/examples/previous", { method: "POST" }); await refreshLibrary(); await loadClip(clip); }
    finally { $("#use-previous").disabled = false; }
  }));
  $("#change-clip").addEventListener("click", newAnalysis);
  $("#new-analysis").addEventListener("click", newAnalysis);
  $("#load-frame").addEventListener("click", () => run(loadPreview));
  $("#clip-start").addEventListener("input", updateReadiness);
  $("#clip-duration").addEventListener("input", updateReadiness);
  $("#confirm-court").addEventListener("change", updateReadiness);
  $("#calibration-reference").addEventListener("change", () => {
    state.points = [];
    $("#confirm-court").checked = false;
    referenceGuide();
    drawCalibration();
  });
  $("#reset-corners").addEventListener("click", () => { state.points = []; $("#confirm-court").checked = false; drawCalibration(); });
  $("#auto-court").addEventListener("click", () => run(async () => {
    if (state.confirmedFrame !== range().start) throw new Error("Actualiza primero el fotograma del inicio.");
    $("#auto-court").disabled = true;
    $("#corner-instruction").textContent = "El modelo está proponiendo la pista…";
    try {
      const proposal = await api(`/api/clips/${state.clip.id}/court?time=${range().start}`);
      if (!proposal.corners) throw new Error(`${proposal.message} Marca las esquinas manualmente o elige otra toma.`);
      $("#calibration-reference").value = "doubles";
      referenceGuide();
      state.points = proposal.corners; $("#confirm-court").checked = false;
      drawCalibration();
      $("#corner-instruction").textContent = proposal.message;
    } finally { $("#auto-court").disabled = false; }
  }));
  const calibration = $("#calibration-canvas");
  const pointer = (event) => {
    const bounds = calibration.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height))];
  };
  calibration.addEventListener("pointerdown", (event) => {
    if (state.confirmedFrame === null) return;
    clearError();
    const point = pointer(event);
    const nearest = state.points.findIndex((existing) => Math.hypot(existing[0] - point[0], existing[1] - point[1]) < 0.07);
    if (nearest >= 0) state.dragging = nearest;
    else if (state.points.length < 4) { state.points.push(point); state.dragging = state.points.length - 1; }
    else { toast("Arrastra uno de los cuatro puntos para ajustarlo."); return; }
    calibration.setPointerCapture(event.pointerId);
    $("#confirm-court").checked = false;
    drawCalibration();
  });
  calibration.addEventListener("pointermove", (event) => {
    if (state.dragging < 0) return;
    state.points[state.dragging] = pointer(event);
    drawCalibration();
  });
  const endDrag = () => { state.dragging = -1; };
  calibration.addEventListener("pointerup", endDrag);
  calibration.addEventListener("pointercancel", endDrag);
  $("#start-analysis").addEventListener("click", () => run(async () => {
    $("#start-analysis").disabled = true;
    try { await startJob(); } finally { updateReadiness(); }
  }));
  $("#cancel-job").addEventListener("click", () => run(async () => {
    if (!active(state.job)) {
      const clip = state.clips.find((item) => item.id === state.job.clip_id);
      if (!clip) throw new Error("El clip ya no está en la biblioteca.");
      await loadClip(clip, state.job.config);
      return;
    }
    state.job = await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" });
    showProgress(state.job);
  }));
  $("#result-video").addEventListener("timeupdate", updateResult);
  $("#result-video").addEventListener("seeked", updateResult);
  $("#result-video").addEventListener("loadedmetadata", updateResult);
  $("#result-video").addEventListener("videoloadstate", () => {
    $("#result-seek").disabled = !resultLoader.ready;
    if (resultLoader.ready) updateResult();
  });
  $("#result-video").addEventListener("error", () => { if (state.result) error("El vídeo no se ha podido reproducir. Prueba a descargar el MP4 y abrirlo con Edge o Chrome."); });
  $("#result-seek").addEventListener("input", () => { $("#result-video").currentTime = Number($("#result-seek").value); updateResult(); });
  $("#coverage-timeline").addEventListener("click", (event) => {
    if (!state.result || !resultLoader.ready) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width * 1200;
    $("#result-video").currentTime = Math.min(state.result.summary.duration, Math.max(0, (x - 80) / 1120 * state.result.summary.duration));
  });
  $$("[data-result-view]").forEach((button) => button.addEventListener("click", () => run(() => switchResultView(button.dataset.resultView))));
  $$("[data-map-mode]").forEach((button) => button.addEventListener("click", () => {
    state.mapMode = button.dataset.mapMode;
    $$("[data-map-mode]").forEach((other) => other.setAttribute("aria-pressed", String(other === button)));
    updateResult();
  }));
  $("#player-filter").addEventListener("change", updateResult);
  $("#copy-storage").addEventListener("click", () => run(async () => {
    await navigator.clipboard.writeText(state.config.storage);
    toast("Ubicación copiada.");
  }));

  run(async () => {
    state.config = await api("/api/config");
    if (state.config.showcase) {
      const example = state.config.showcase;
      const originalLoader = new window.TennisVideoLoader($("#showcase-original"));
      const analysisLoader = new window.TennisVideoLoader($("#showcase-analysis"));
      void originalLoader.load(`/api/jobs/${example.job_id}/files/original.mp4`);
      void analysisLoader.load(`/api/jobs/${example.job_id}/files/analysis.mp4`);
      $("#showcase-analysis").poster = `/api/jobs/${example.job_id}/files/poster.jpg`;
      if (example.title) $("#showcase-title").textContent = example.title;
      $("#showcase-credit").textContent = example.credit;
      $("#showcase-notes").replaceChildren(...example.notes.map((note) => element("li", note)));
      $("#showcase").hidden = false;
    }
    $("#engine-status").textContent = state.config.models.people ? "Motor de jugadores preparado" : "Faltan modelos: ejecuta la preparación";
    $("#storage-path").textContent = state.config.storage;
    $("#use-previous").hidden = !state.config.previous_demo_available;
    $("#complete-mode").disabled = !state.config.models.ball && !state.config.models.ball_yolo;
    $("#ball-backend option[value=tracknet]").disabled = !state.config.models.ball;
    $("#ball-backend option[value=yolo]").disabled = !state.config.models.ball_yolo;
    $("#ball-backend option[value=yolo_multiscale]").disabled = !state.config.models.ball_yolo;
    $("#ball-backend option[value=yolo_wasb]").disabled = !(state.config.models.ball_yolo && state.config.models.ball_wasb);
    $("#ball-backend").value = state.config.models.ball_yolo ? "yolo_multiscale" : "tracknet";
    if ($("#complete-mode").disabled) $("#players-mode").checked = true;
    $("#ball-backend").addEventListener("change", updateReadiness);
    $$("input[name=analysis-mode]").forEach((input) => input.addEventListener("change", updateReadiness));
    $("#auto-court").disabled = !state.config.models.court;
    $("#model-list").replaceChildren();
    for (const [key, label] of [["people", "YOLOX-S · Personas"], ["court", "ResNet50 · Propuesta de pista"], ["ball", "TrackNet · Pelota experimental"], ["ball_yolo", "YOLOv8n tenis · Pelota ligera experimental"], ["ball_wasb", "WASB HRNet · Pelota multifotograma (pesos locales, licencia no aclarada)"]]) {
      $("#model-list").append(element("li", `${label}: ${state.config.models[key] ? "instalado" : "no disponible"}`));
    }
    await refreshLibrary();
    stage("upload");
    setInterval(poll, 1200);
    setInterval(() => { if (state.screen === "library") run(refreshLibrary); }, 5000);
  });
})();
