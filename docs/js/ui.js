/* Small UI touches on top of Material: a reading-progress bar and a
   reading-time chip under each note's title. Everything re-runs after
   Material's instant navigation via document$. */
(function () {
  "use strict";

  var CLOCK_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20a8 8 0 0 0 8-8 8 8 0 0 0-8-8 8 8 0 0 0-8 8 8 8 0 0 0 8 8m0-18a10 10 0 0 1 10 10 10 10 0 0 1-10 10C6.47 22 2 17.5 2 12A10 10 0 0 1 12 2m.5 5v5.25l4.5 2.67-.75 1.23L11 13V7z"/></svg>';

  /* Reading progress ------------------------------------------------------ */

  var bar = document.createElement("div");
  bar.className = "wk-progress";
  bar.setAttribute("aria-hidden", "true");

  var ticking = false;
  function paintProgress() {
    ticking = false;
    var root = document.documentElement;
    var max = root.scrollHeight - root.clientHeight;
    var ratio = max > 0 ? Math.min(1, root.scrollTop / max) : 0;
    bar.style.transform = "scaleX(" + ratio + ")";
  }
  function requestProgress() {
    if (!ticking) {
      ticking = true;
      window.requestAnimationFrame(paintProgress);
    }
  }
  window.addEventListener("scroll", requestProgress, { passive: true });
  window.addEventListener("resize", requestProgress);

  /* Reading time ---------------------------------------------------------- */

  function addReadingTime() {
    var article = document.querySelector(".md-content__inner");
    if (!article || article.querySelector(".wk-home, .wk-meta")) return;
    var title = article.querySelector("h1");
    if (!title) return;

    var copy = article.cloneNode(true);
    copy.querySelectorAll(".arithmatex, .headerlink, .md-source-file, script").forEach(function (el) {
      el.remove();
    });
    var words = (copy.textContent.match(/\S+/g) || []).length;
    var equations = article.querySelectorAll("div.arithmatex").length;
    // ~220 wpm for prose, plus a little time to actually read each displayed equation
    var minutes = Math.max(1, Math.round(words / 220 + equations * 0.25));

    var meta = document.createElement("div");
    meta.className = "wk-meta";
    meta.innerHTML = '<span class="wk-chip">' + CLOCK_ICON + minutes + " min read</span>";
    title.insertAdjacentElement("afterend", meta);
  }

  /* Wiring ---------------------------------------------------------------- */

  function onPage() {
    if (!bar.isConnected) document.body.appendChild(bar);
    addReadingTime();
    requestProgress();
  }

  if (window.document$) {
    window.document$.subscribe(onPage);
  } else {
    document.addEventListener("DOMContentLoaded", onPage);
  }
})();
