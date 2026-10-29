window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    tags: "all",
    tagSide: "right"
  },
  options: {
    processHtmlClass: "arithmatex",
    ignoreHtmlClass: ".*"
  }
};

// With instant navigation, pages are swapped in without a full reload, so
// re-typeset (and reset equation numbering) each time a new page arrives.
// The first page is typeset by MathJax's own startup.
var firstPage = true;
document$.subscribe(function () {
  if (firstPage) {
    firstPage = false;
    return;
  }
  if (!window.MathJax || !MathJax.startup || !MathJax.startup.promise) return;
  MathJax.startup.promise = MathJax.startup.promise.then(function () {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
