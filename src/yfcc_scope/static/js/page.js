(function () {
  const form = document.getElementById("ctrl_form");
  const selectAll = document.getElementById("select_all");
  const grid = document.getElementById("labels_grid");
  const hiddenAll = document.getElementById("select_all_hidden");
  const minConf = document.getElementById("min_conf");
  const minConfVal = document.getElementById("min_conf_val");

  function updateSelectAllState() {
    const boxes = Array.from(
      grid.querySelectorAll('input[type="checkbox"][name="label"]'),
    );
    const checked = boxes.filter((b) => b.checked).length;
    const isAll = checked === boxes.length;
    selectAll.checked = isAll;
    hiddenAll.value = isAll ? "1" : "0";
  }

  if (form) {
    form.addEventListener("submit", () => {
      const boxes = grid.querySelectorAll(
        'input[type="checkbox"][name="label"]',
      );
      if (selectAll.checked) {
        boxes.forEach((b) => (b.disabled = true));
        setTimeout(() => boxes.forEach((b) => (b.disabled = false)), 0);
      }
    });
  }

  if (selectAll) {
    selectAll.addEventListener("change", () => {
      const boxes = grid.querySelectorAll(
        'input[type="checkbox"][name="label"]',
      );
      if (selectAll.checked) {
        boxes.forEach((b) => (b.checked = true));
        hiddenAll.value = "1";
      } else {
        boxes.forEach((b) => (b.checked = false));
        hiddenAll.value = "0";
      }
      form.requestSubmit();
    });
  }

  if (grid) {
    grid.addEventListener("change", (e) => {
      if (e.target && e.target.name === "label") {
        updateSelectAllState();
        form.requestSubmit();
      }
    });
  }

  function updateMinConfVal() {
    if (minConfVal) minConfVal.textContent = Number(minConf.value).toFixed(2);
  }

  if (minConf) {
    minConf.addEventListener("input", updateMinConfVal);
    minConf.addEventListener("change", () => {
      form.requestSubmit();
    });
  }

  if (grid && selectAll && hiddenAll) {
    updateSelectAllState();
  }

  if (minConf && minConfVal) {
    updateMinConfVal();
  }
})();
