"use strict";
const $ = (id) => document.getElementById(id);
const SAMPLES = [
  "Dear customer, your SBI account will be blocked today. Update your KYC immediately: http://sbi-kyc-update.xyz/a8Xk2",
  "Congratulations! You have won Rs 25,00,000 in the Jio lucky draw. Pay Rs 999 registration fee to rahul22@ybl to claim your prize",
  "Your refund of Rs 4,999 is ready. Approve the collect request on GPay and enter your UPI PIN to receive it",
  "Your Aadhaar is linked to a money laundering case. Call officer on 9876543210 immediately to avoid arrest. Do not tell anyone.",
  "Aapka HDFC account aaj band ho jayega. Turant KYC update karein: http://bit.ly/kyc9Xz",
  "Rs 2,500 debited from A/c XX4821 on 20-11-2026 at Swiggy. Avl bal Rs 18,200. Not you? Call the number on the back of your card.",
  "Your OTP for transaction of Rs 1,200 at Amazon is 482913. Do not share this OTP with anyone. -HDFC Bank",
];const DEMO = [SAMPLES[0], SAMPLES[1], SAMPLES[3], SAMPLES[4], SAMPLES[5], SAMPLES[6],
  "Hi, are we still meeting for coffee at 5?", "Your Amazon order has shipped. Track at https://www.amazon.in/gp/your-orders"];
const CIRC = 2 * Math.PI * 52;
let last = null, timer = null, seq = 0, batchRows = [];
let modelId = "lr";
const SHORT = { lr: "Logistic Regression", bigru: "BiGRU", distilbert: "DistilBERT" };

/* ---------- storage (guarded: may be unavailable) ---------- */
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
};

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + t.dataset.view));
  if (t.dataset.view === "model") loadModel();
}));

/* ---------- api ---------- */
async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed (" + res.status + ")");
  return data;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/* ---------- scan ---------- */
const msg = $("msg");
msg.addEventListener("input", () => {
  $("count").textContent = msg.value.length + " / 5000";
  clearTimeout(timer);
  timer = setTimeout(check, 400);
});
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { msg.value = SAMPLES[+c.dataset.s]; msg.dispatchEvent(new Event("input")); check(true); }));
$("clear").addEventListener("click", () => { msg.value = ""; msg.dispatchEvent(new Event("input")); msg.focus(); });

async function check(save) {
  const text = msg.value.trim(), my = ++seq;
  if (!text) { $("result").hidden = true; last = null; return; }
  try {
    const r = await post("/api/check", { text, model: modelId });
    if (my !== seq) return;           // a newer request superseded this one
    last = { text, r };
    render(r);
    if (save === true) addHistory(text, r);
    else { clearTimeout(check.h); check.h = setTimeout(() => { if (my === seq) addHistory(text, r); }, 2500); }
  } catch (e) { showError(e.message); }
}

function showError(m) {
  const el = $("result"); el.hidden = false; el.className = "card result";
  $("badge").textContent = "Error"; $("res-title").textContent = "Something went wrong"; $("res-sub").textContent = m;
}

function render(r) {
  const el = $("result");
  el.hidden = false;
  el.className = "card result " + r.verdict;
  $("badge").textContent = r.label;
  $("g-num").textContent = r.risk_score;
  $("g-fg").style.strokeDashoffset = CIRC * (1 - r.probability);
  const titles = { scam: "This looks like a scam", suspicious: "Be careful with this message", safe: "No strong scam signs found" };
  $("res-title").textContent = titles[r.verdict];
  $("res-sub").textContent = r.scam_type ? "Looks like: " + r.scam_type : (r.verdict === "safe" ? "Matches patterns of genuine messages." : "");
  const m = $("marked"); m.textContent = "";
  r.segments.forEach((s) => {
    if (s.level === "none") { m.appendChild(document.createTextNode(s.text)); return; }
    const sp = document.createElement("span");
    sp.className = s.level; sp.textContent = s.text;
    sp.title = (s.score > 0 ? "+" : "") + s.score.toFixed(1) + " risk contribution";
    m.appendChild(sp);
  });
  const sg = $("signals"); sg.textContent = "";
  const reasons = r.reasons.length ? r.reasons : [{ kind: "none", text: "No known scam patterns were detected in this message." }];
  reasons.forEach((x) => { const li = document.createElement("li"); li.className = x.kind; li.textContent = x.text; sg.appendChild(li); });
  const ad = $("advice"); ad.textContent = "";
  r.advice.forEach((x) => { const li = document.createElement("li"); li.textContent = x; ad.appendChild(li); });
  $("fb-msg").textContent = "";
  $("fb-ok").disabled = $("fb-wrong").disabled = false;
  renderVotes(r);
  $("xp-out").textContent = ""; $("xp-note").textContent = "Pick a method to see which words and signals pushed the score.";
}

function renderVotes(r) {
  const keys = Object.keys(r.votes || {});
  $("votes-wrap").hidden = keys.length < 2;
  const box = $("votes"); box.textContent = "";
  keys.forEach((k) => {
    const p = r.votes[k], row = document.createElement("div");
    row.className = "vote" + (k === r.model ? " sel" : "");
    const name = document.createElement("span"); name.textContent = SHORT[k] || k;
    const tr = document.createElement("div"); tr.className = "track";
    const f = document.createElement("div"); f.className = "fill"; f.style.width = (p * 100) + "%";
    f.style.background = p >= 0.65 ? "var(--bad)" : p >= 0.35 ? "var(--warn)" : "var(--safe)";
    tr.appendChild(f);
    const v = document.createElement("b"); v.textContent = Math.round(p * 100) + "%";
    row.append(name, tr, v); box.appendChild(row);
  });
}

/* ---------- feedback ---------- */
async function sendFeedback(correct) {
  if (!last) return;
  const predictedScam = last.r.verdict !== "safe";
  const label = correct ? (predictedScam ? "scam" : "legit") : (predictedScam ? "legit" : "scam");
  try {
    await post("/api/feedback", { text: last.text, label, predicted: last.r.verdict });
    $("fb-msg").textContent = "Thanks! Saved for retraining.";
    $("fb-ok").disabled = $("fb-wrong").disabled = true;
  } catch (e) { $("fb-msg").textContent = e.message; }
}
$("fb-ok").addEventListener("click", () => sendFeedback(true));
$("fb-wrong").addEventListener("click", () => sendFeedback(false));

/* ---------- history ---------- */
function renderHistory() {
  const h = store.get("ss-history", []), ul = $("history");
  ul.textContent = "";
  $("hist-empty").hidden = h.length > 0;
  h.forEach((it) => {
    const li = document.createElement("li");
    const dot = document.createElement("span"); dot.className = "dot " + it.v;
    const t = document.createElement("span"); t.className = "t"; t.textContent = it.text;
    const s = document.createElement("span"); s.className = "s"; s.textContent = it.score;
    li.append(dot, t, s);
    li.addEventListener("click", () => { msg.value = it.text; msg.dispatchEvent(new Event("input")); check(); window.scrollTo({ top: 0, behavior: "smooth" }); });
    ul.appendChild(li);
  });
}
function addHistory(text, r) {
  const h = store.get("ss-history", []).filter((x) => x.text !== text);
  h.unshift({ text, v: r.verdict, score: r.risk_score });
  store.set("ss-history", h.slice(0, 8));
  renderHistory();
}
$("clear-hist").addEventListener("click", () => { store.set("ss-history", []); renderHistory(); });

/* ---------- batch ---------- */
const drop = $("drop");
["dragover", "dragenter"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => { if (ev.dataTransfer.files[0]) runBatchFile(ev.dataTransfer.files[0]); });
$("file").addEventListener("change", (ev) => { if (ev.target.files[0]) runBatchFile(ev.target.files[0]); });
$("demo-batch").addEventListener("click", async () => { try { showBatch(await post("/api/batch", { texts: DEMO })); } catch (e) { alert(e.message); } });

async function runBatchFile(file) {
  const fd = new FormData(); fd.append("file", file);
  try { showBatch(await api("/api/batch", { method: "POST", body: fd })); } catch (e) { alert(e.message); }
}
function showBatch(d) {
  batchRows = d.rows;
  $("batch-res").hidden = false;
  const s = d.summary, sum = $("sum"); sum.textContent = "";
  [["Messages", s.total], ["Scam", s.scam], ["Suspicious", s.suspicious], ["Safe", s.safe]].forEach(([k, v]) => {
    const dv = document.createElement("div"), b = document.createElement("b"), sp = document.createElement("span");
    b.textContent = v; sp.textContent = k; dv.append(b, sp); sum.appendChild(dv);
  });
  if (d.truncated) { const n = document.createElement("div"); n.className = "notice"; n.textContent = "Only the first " + d.limit + " messages were scored."; sum.after(n); setTimeout(() => n.remove(), 8000); }
  const tb = $("tbody"); tb.textContent = "";
  d.rows.forEach((r) => {
    const tr = document.createElement("tr");
    const cells = [r.index, r.text, r.risk_score + "/100", null, r.scam_type || "—"];
    cells.forEach((c, i) => {
      const td = document.createElement("td");
      if (i === 1) td.className = "msg";
      if (i === 3) { const p = document.createElement("span"); p.className = "pill " + r.verdict; p.textContent = r.label; td.appendChild(p); }
      else td.textContent = c;
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
}
$("dl").addEventListener("click", () => {
  const esc = (v) => '"' + String(v ?? "").replace(/"/g, '""') + '"';
  const csv = ["index,text,risk_score,verdict,scam_type"].concat(batchRows.map((r) => [r.index, esc(r.text), r.risk_score, r.verdict, esc(r.scam_type)].join(","))).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "scamshield_results.csv"; a.click(); URL.revokeObjectURL(a.href);
});

/* ---------- model card ---------- */
let modelLoaded = false;
async function loadModel() {
  if (modelLoaded) return;
  const root = $("model-root");
  try {
    const c = await api("/api/model");
    modelLoaded = true; root.textContent = "";
    const pct = (v) => (v * 100).toFixed(1) + "%";
    const t = c.metrics.test, f = c.metrics.flagged;
    if (!c.data.real_data_used) {
      const n = document.createElement("div"); n.className = "notice";
      n.textContent = "This model was trained on synthetic data only, so these scores are optimistic. Run python -m scamshield.download_data and retrain to add real SMS data.";
      root.appendChild(n);
    }
    const k = document.createElement("div"); k.className = "kpis";
    [["Precision", pct(t.precision), "of 'scam' verdicts were correct"], ["Recall", pct(t.recall), "of scams caught as 'scam'"],
     ["Caught or flagged", pct(f.recall), "scams flagged scam or suspicious"], ["F1 score", t.f1.toFixed(3), "balance of both"], ["ROC-AUC", t.roc_auc.toFixed(3), "ranking quality"]]
      .forEach(([l, v, s]) => { const d = document.createElement("div"); d.className = "kpi"; d.title = s; const b = document.createElement("b"), sp = document.createElement("span"); b.textContent = v; sp.textContent = l; d.append(b, sp); k.appendChild(d); });
    root.appendChild(k);

    const info = document.createElement("div"); info.className = "card";
    const h = document.createElement("h3"); h.style.marginTop = 0; h.textContent = "Training summary"; info.appendChild(h);
    const p = document.createElement("p"); p.style.margin = 0;
    const src = Object.entries(c.data.by_source).map(([a, b]) => a + " (" + b.toLocaleString() + ")").join(", ");
    p.textContent = c.algorithm + ". Trained on " + c.data.train.toLocaleString() + " messages, tested on " + c.data.test.toLocaleString() +
      " unseen ones. Data sources: " + src + ". Alert thresholds chosen by cross-validation: suspicious at " + c.thresholds.suspicious +
      ", scam at " + c.thresholds.scam + ". Version " + c.version + ", trained " + c.trained_at.slice(0, 10) + ".";
    info.appendChild(p); root.appendChild(info);

    const cmp = document.createElement("div"); cmp.className = "card";
    const h2 = document.createElement("h3"); h2.style.marginTop = 0; h2.textContent = "Model comparison (held-out test set)"; cmp.appendChild(h2);
    const tw = document.createElement("div"); tw.className = "table-wrap";
    const tbl = document.createElement("table"); tbl.className = "tbl";
    const hd = tbl.createTHead().insertRow();
    ["Model", "Precision", "Recall", "F1", "ROC-AUC"].forEach((x) => { const th = document.createElement("th"); th.textContent = x; hd.appendChild(th); });
    const bd = tbl.createTBody();
    c.model_comparison.forEach((m) => {
      const tr = bd.insertRow();
      [m.model, m.precision, m.recall, m.f1, m.roc_auc].forEach((x, i) => { const td = tr.insertCell(); td.textContent = i ? Number(x).toFixed(3) : x; if (m.model.includes("ScamShield")) td.style.fontWeight = 700; });
    });
    tw.appendChild(tbl); cmp.appendChild(tw); root.appendChild(cmp);

    try {
      const { models } = await api("/api/models");
      const dc = document.createElement("div"); dc.className = "card";
      const dh = document.createElement("h3"); dh.style.marginTop = 0; dh.textContent = "Models in this build"; dc.appendChild(dh);
      const list = document.createElement("div"); list.className = "deep-status";
      models.forEach((m) => {
        const r = document.createElement("div");
        const a = document.createElement("span"); a.textContent = m.label;
        const b = document.createElement("span"); b.className = "pill " + (m.available ? "safe" : "off"); b.textContent = m.available ? "Ready" : m.message;
        r.append(a, b); list.appendChild(r);
      });
      dc.appendChild(list); root.appendChild(dc);
    } catch (e) { /* ignore */ }

    const pl = document.createElement("div"); pl.className = "plots";
    [["model_comparison", 1], ["roc_pr", 1], ["confusion_matrix", 0], ["calibration", 0], ["shap_importance", 1], ["shap_beeswarm", 1], ["top_features", 1]].filter(([n]) => (c.plots || []).includes(n)).forEach(([n, wide]) => {
      const card = document.createElement("div"); card.className = "card" + (wide ? " wide" : "");
      const img = document.createElement("img"); img.src = "/plots/" + n + ".png"; img.alt = n.replace(/_/g, " ") + " chart"; img.loading = "lazy";
      img.addEventListener("error", () => card.remove());        // optional plots (e.g. SHAP beeswarm) may not exist
      card.appendChild(img); pl.appendChild(card);
    });
    root.appendChild(pl);

    const lim = document.createElement("div"); lim.className = "card";
    const h3 = document.createElement("h3"); h3.style.marginTop = 0; h3.textContent = "Known limitations"; lim.appendChild(h3);
    const ul = document.createElement("ul"); ul.style.margin = 0;
    c.limitations.forEach((x) => { const li = document.createElement("li"); li.textContent = x; ul.appendChild(li); });
    lim.appendChild(ul); root.appendChild(lim);
  } catch (e) { root.textContent = ""; const n = document.createElement("div"); n.className = "notice"; n.textContent = e.message; root.appendChild(n); }
}

/* ---------- model selector ---------- */
async function loadModels() {
  try {
    const { models } = await api("/api/models");
    const sel = $("model"); sel.textContent = "";
    models.forEach((m) => {
      const o = document.createElement("option");
      o.value = m.id; o.textContent = m.label + (m.available ? "" : " (not available)"); o.disabled = !m.available; o.title = m.message;
      sel.appendChild(o);
    });
    const saved = store.get("ss-model", "lr");
    modelId = models.some((m) => m.id === saved && m.available) ? saved : "lr";
    sel.value = modelId;
    const off = models.filter((m) => !m.available && m.id !== "lr");
    $("model-note").textContent = off.length ? "Deep models: " + off.map((m) => SHORT[m.id] || m.label).join(", ") + " not set up yet (see Model & accuracy tab)." : "All models ready.";
  } catch (e) { /* keep default */ }
}
$("model").addEventListener("change", (e) => { modelId = e.target.value; store.set("ss-model", modelId); check(); });

/* ---------- explanations (SHAP / LIME) ---------- */
function bars(rows) {
  const box = $("xp-out"); box.textContent = "";
  const max = Math.max(0.001, ...rows.map((r) => Math.abs(r.value)));
  rows.forEach((r) => {
    const row = document.createElement("div"); row.className = "xrow";
    const lab = document.createElement("span"); lab.className = "lab"; lab.textContent = r.label; lab.title = r.label;
    const bar = document.createElement("div"); bar.className = "bar";
    const i = document.createElement("i"); i.className = r.value >= 0 ? "pos" : "neg"; i.style.width = (Math.abs(r.value) / max * 50) + "%";
    bar.appendChild(i);
    const v = document.createElement("b"); v.textContent = (r.value > 0 ? "+" : "") + r.value.toFixed(2);
    row.append(lab, bar, v); box.appendChild(row);
  });
}
async function explainWith(method) {
  if (!last) return;
  const model = method === "shap" ? "lr" : modelId;
  $("xp-note").textContent = method === "lime" && model !== "lr" ? "LIME is re-running the model on word-removed copies: this can take a few seconds…" : "Working…";
  $("xp-out").textContent = "";
  try {
    const d = await post("/api/explain", { text: last.text, method, model });
    if (method === "shap") {
      $("xp-note").textContent = "SHAP values for the Logistic Regression model, compared with an average training message (score " + d.base_logit.toFixed(1) +
        " → " + d.logit.toFixed(1) + " log-odds). Red pushes toward scam, green toward genuine.";
      bars(d.contributions.map((c) => ({ label: c.feature, value: c.shap })));
    } else {
      $("xp-note").textContent = "LIME on " + (SHORT[model] || model) + ": weight of each word toward 'scam' (red) or 'genuine' (green).";
      bars(d.contributions.map((c) => ({ label: c.word, value: c.weight })));
    }
  } catch (e) { $("xp-note").textContent = e.message; }
}
$("xp-shap").addEventListener("click", () => explainWith("shap"));
$("xp-lime").addEventListener("click", () => explainWith("lime"));

/* ---------- init ---------- */
renderHistory();
loadModels();
api("/api/health").then((h) => { $("ver").textContent = "v" + h.version; }).catch(() => {});
