(function () {
  async function getJSON(u) {
    const r = await fetch(u, { cache: "no-store" });
    if (!r.ok) throw new Error("fetch " + u);
    return r.json();
  }
  async function post(u) {
    const r = await fetch(u, { method: "POST" });
    if (!r.ok) throw new Error("post " + u);
    return r.text();
  }

  let freqMap = {};
  let updatedAt = 0;
  let totalImagesYFCC = 0;
  let imagesWithBoxes = 0;
  let originalOrder = null;

  function labelKey(lbl) {
    const input = lbl.querySelector('input[name="label"]');
    return input ? input.value : lbl.textContent.trim();
  }

  function decorateFractions(grid) {
    grid.querySelectorAll("label.lbl").forEach((lbl) => {
      const key = labelKey(lbl);
      const ent = freqMap[key] || { fraction: 0 };
      const pretty = `(${ent.fraction.toFixed(4)})`;
      let span = lbl.querySelector("span.__freq");
      if (!span) {
        span = document.createElement("span");
        span.className = "muted __freq";
        span.style.marginLeft = "4px";
        lbl.appendChild(span);
      }
      span.textContent = pretty;
    });
  }

  function updateFooter() {
    const note = document.getElementById("freq_updated_note");
    if (!note) return;
    if (updatedAt) {
      const ts = new Date(updatedAt * 1000);
      note.textContent =
        "Frequencies last updated: " +
        ts.toLocaleString() +
        " — Total images in index: " +
        totalImagesYFCC.toLocaleString() +
        " — Images with boxes (at threshold): " +
        imagesWithBoxes.toLocaleString();
    } else {
      note.textContent = "";
    }
  }

  function sortGridBy(grid, cmp) {
    const items = Array.from(grid.querySelectorAll("label.lbl"));
    items.sort(cmp);
    items.forEach((el) => grid.appendChild(el));
  }

  async function main() {
    const grid = document.getElementById("labels_grid");
    const selectAllRow = document.getElementById("top_toolbar");
    const minConf = document.getElementById("min_conf");
    if (!grid || !minConf) return;
    if (!originalOrder)
      originalOrder = Array.from(grid.querySelectorAll("label.lbl"));

    const btnOrig = document.getElementById("sort_original");
    const btnAlpha = document.getElementById("sort_alpha");
    const btnFreq = document.getElementById("sort_freq");

    btnOrig?.addEventListener("click", () =>
      originalOrder.forEach((el) => grid.appendChild(el)),
    );
    btnAlpha?.addEventListener("click", () =>
      sortGridBy(grid, (a, b) => labelKey(a).localeCompare(labelKey(b))),
    );
    btnFreq?.addEventListener("click", () =>
      sortGridBy(grid, (a, b) => {
        const fa = freqMap[labelKey(a)]?.fraction || 0,
          fb = freqMap[labelKey(b)]?.fraction || 0;
        if (fb !== fa) return fb - fa;
        return labelKey(a).localeCompare(labelKey(b));
      }),
    );

    if (selectAllRow && !document.getElementById("recalc_btn")) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.id = "recalc_btn";
      btn.textContent = "Recalculate frequencies";
      btn.style.marginLeft = "12px";
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        btn.textContent = "Recalculating...";
        try {
          await post("/recalc");
          location.reload();
        } catch (e) {
          alert("Recalc failed: " + e.message);
          btn.disabled = false;
          btn.textContent = "Recalculate frequencies";
        }
      });
      selectAllRow.appendChild(btn);
    }

    let data;
    try {
      const c = Number(minConf.value).toFixed(2);
      data = await getJSON("/api/freqs?min_conf=" + encodeURIComponent(c));
    } catch (e) {
      return;
    }
    freqMap = data.labels || {};
    updatedAt = data.updated_at || 0;
    totalImagesYFCC = data.total_images_yfcc || 0;
    imagesWithBoxes = data.images_with_boxes || 0;

    decorateFractions(grid);
    updateFooter();
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", main);
  else main();
})();
