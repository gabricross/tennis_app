"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const navigation = [...document.querySelectorAll(".sidebar nav a[href^='#']")];
  const updateNavigation = () => {
    const current = window.location.hash || "#examples";
    for (const link of navigation) {
      if (link.hash === current) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  };
  updateNavigation();
  window.addEventListener("hashchange", updateNavigation);

  for (const type of ["dragover", "drop"]) {
    document.addEventListener(type, (event) => {
      if (!event.dataTransfer?.types.includes("Files")) return;
      event.preventDefault();
      if (type === "drop") {
        const notice = document.getElementById("demo-upload-notice");
        notice.hidden = false;
        notice.scrollIntoView({ block: "center" });
      }
    });
  }
});
