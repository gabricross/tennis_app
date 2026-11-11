"use strict";

window.TennisExamples = {
  render(container, examples) {
    container.replaceChildren();
    for (const [index, example] of examples.entries()) {
      const identity = `${container.id}-${index}`;
      const section = document.createElement("section");
      section.className = "media-example showcase panel";
      section.setAttribute("aria-labelledby", `${identity}-title`);
      section.innerHTML = `
        <h2></h2>
        <div class="video-pair-controls" role="group" aria-label="Reproducción de este ejemplo">
          <button type="button" data-video-action="play" disabled><span aria-hidden="true">▶</span> Reproducir ambos</button>
          <button type="button" data-video-action="pause" disabled>Pausar</button>
          <button type="button" data-video-action="restart" disabled>Volver al inicio</button>
          <p data-pair-error role="alert" hidden></p>
        </div>
        <div class="showcase-grid">
          <figure><figcaption>Original</figcaption><video hidden playsinline preload="none" aria-label="Vídeo original de este ejemplo"></video></figure>
          <figure><figcaption>Análisis</figcaption><video hidden playsinline preload="none" aria-label="Análisis de este ejemplo"></video></figure>
        </div>
        <p class="small-note example-credit"></p>
        <p class="small-note example-modifications"></p>
        <ul class="small-note"></ul>`;
      const title = section.querySelector("h2");
      title.id = `${identity}-title`;
      title.textContent = example.title || "Original y análisis";
      const videos = [...section.querySelectorAll("video")];
      videos[0].id = `${identity}-original`;
      videos[1].id = `${identity}-analysis`;
      section.querySelector("[data-video-action]").parentElement.dataset.videoPair = videos.map((video) => video.id).join(" ");
      const credit = section.querySelector(".example-credit");
      credit.textContent = example.credit;
      for (const [url, label] of [[example.source_url, "Fuente"], [example.license_url, example.license]]) {
        if (!url) continue;
        if (new URL(url).protocol !== "https:") throw new Error("Los créditos del ejemplo necesitan enlaces HTTPS.");
        const link = document.createElement("a");
        link.href = url;
        link.textContent = label;
        credit.append(document.createTextNode(" · "), link);
      }
      const modifications = section.querySelector(".example-modifications");
      modifications.textContent = example.modifications || "";
      modifications.hidden = !example.modifications;
      for (const note of example.notes) {
        const item = document.createElement("li");
        item.textContent = note;
        section.querySelector("ul").append(item);
      }
      container.append(section);
      const loaders = videos.map((video) => new window.TennisVideoLoader(video));
      setupVideoPair(section.querySelector("[data-video-pair]"));
      void loaders[0].load(example.original);
      void loaders[1].load(example.analysis);
    }
  },
};

document.addEventListener("DOMContentLoaded", async () => {
  const container = document.querySelector("[data-example-manifest]");
  if (!container) return;
  try {
    const response = await fetch(container.dataset.exampleManifest);
    if (!response.ok) throw new Error(`No se han podido cargar los ejemplos (${response.status}).`);
    window.TennisExamples.render(container, await response.json());
  } catch (failure) {
    console.error("Ejemplos de vídeo:", failure);
    const error = document.createElement("p");
    error.setAttribute("role", "alert");
    error.textContent = `No se han podido mostrar los ejemplos. Recarga la página: ${failure.message}`;
    container.append(error);
  }
});
