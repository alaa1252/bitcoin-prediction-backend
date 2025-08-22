// ============================
// Config & State
// ============================
const API_BASE = "http://127.0.0.1:5000";  // Flask backend
let autoRefresh = true;
let chart;
let currentCurrency = "USD";

const state = {
  mockMode: false,
  mockSeed: 12345
};


const RATES = {
    USD: 1, 
    ILS: 3.7, //fixed first, then dynamic
    JOD: 0.71  
};

const SYMBOLS = {
  USD: "$",
  ILS: "₪",
  JOD: "JD"
};

// ============================
// Helpers
// ============================
function fmt(num, cur) {
  return SYMBOLS[cur] + num.toFixed(2);
}

function updateAsOf() {
  const now = new Date();
  document.getElementById("asOfTime").textContent = now.toLocaleString();
}

// ============================
// API Calls
// ============================
async function fetchHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    return await r.json();
  } catch (e) {
    console.error("Health check failed", e);
    return null;
  }
}

async function fetchPredict() {
  try {
    const r = await fetch(`${API_BASE}/predict`);
    return await r.json();
  } catch (e) {
    console.error("Predict failed", e);
    return null;
  }
}

async function fetchHistorical() {
  try {
    const r = await fetch(`${API_BASE}/historical`);
    return await r.json();
  } catch (e) {
    console.error("Historical failed", e);
    return null;
  }
}

async function updateRates() {
  try {
    // Fetch ILS and JOD relative to USD
    const res = await fetch("https://api.exchangerate.host/latest?base=USD&symbols=ILS,JOD");
    const data = await res.json();
    
    if (data && data.rates) {
      RATES.ILS = data.rates.ILS;
      RATES.JOD = data.rates.JOD;
      console.log("Updated exchange rates:", RATES);
    }
  } catch (err) {
    console.error("Failed to fetch exchange rates:", err);
  }
}

// ============================
// UI Updates
// ============================
function updateCards(data) {
  if (!data || data.status !== "success") return;

  ["USD", "ILS", "JOD"].forEach(cur => {
    const rate = RATES[cur];
    const nowVal = data.current_price * rate;
    const predVal = data.predicted_price * rate;
    const deltaPct = ((predVal - nowVal) / nowVal) * 100;

    document.getElementById(cur.toLowerCase() + "Now").textContent = fmt(nowVal, cur);
    document.getElementById(cur.toLowerCase() + "Pred").textContent = fmt(predVal, cur);

    const deltaEl = document.getElementById(cur.toLowerCase() + "Delta");
    deltaEl.textContent = deltaPct.toFixed(2) + "%";
    deltaEl.style.color = deltaPct >= 0 ? "green" : "red";
  });
}

function updateChart(hist, pred, cur) {
  if (!hist || hist.status !== "success") return;

  const rate = RATES[cur];

  // Chart labels and actual prices
  const labels = hist.timestamps.map(ts => new Date(ts).toLocaleDateString());
  const prices = hist.prices.map(p => p * rate);

  // Predicted dataset: actual + predicted as separate last point
  const predictedPrices = [...prices.slice(0, -1), pred.predicted_price * rate];

  if (chart) chart.destroy();

  const ctx = document.getElementById("btcChart").getContext("2d");
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual",
          data: prices,
          borderColor: "blue",
          backgroundColor: "rgba(0,0,255,0.1)",
          tension: 0.2,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: "Predicted",
          data: predictedPrices,
          borderColor: "orange",
          backgroundColor: "rgba(255,165,0,0.1)",
          borderDash: [5, 5],
          tension: 0.2,
          pointRadius: 4,
          pointHoverRadius: 6,
        }
      ]
    },
    options: {
      responsive: true,
      interaction: {
        mode: "nearest",
        intersect: false,
      },
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: function(context) {
              return context.dataset.label + ": " + SYMBOLS[cur] + context.raw.toFixed(2);
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: false
        }
      }
    }
  });

  // Update title + unit
  document.getElementById("seriesUnit").textContent = SYMBOLS[cur];
  document.getElementById("chartTitle").textContent =
    cur === "USD"
      ? "BTC vs US Dollar — Actual vs Predicted"
      : cur === "ILS"
      ? "BTC vs Shekel — Actual vs Predicted"
      : "BTC vs Jordanian Dinar — Actual vs Predicted";
}

// ============================
// Mock Data Generator
// ============================
function generateMockData() {
  const basePrice = 30000 + Math.random() * 5000;
  const predPrice = basePrice * (1 + (Math.random() - 0.5) / 10);

  const timestamps = Array.from({ length: 20 }, (_, i) =>
    new Date(Date.now() - (19 - i) * 24 * 3600 * 1000).toISOString()
  );

  const prices = timestamps.map((_, i) => basePrice * (1 + (Math.random() - 0.5) / 20 * i));

  return {
    pred: {
      status: "success",
      current_price: basePrice,
      predicted_price: predPrice
    },
    hist: {
      status: "success",
      prices,
      timestamps
    }
  };
}

// ============================
// Main Refresh
// ============================
async function refreshAll() {
   
    updateAsOf();
    await updateRates();
     // make sure exchange rates are updated

    let pred, hist;

    if (state.mockMode) {
        const mock = generateMockData();
        pred = mock.pred;
        hist = mock.hist;
    } else {
        pred = await fetchPredict();
        hist = await fetchHistorical();
    }

  if (pred) updateCards(pred);
  if (hist && pred) updateChart(hist, pred, currentCurrency);
}

// ============================
// Event Listeners
// ============================
document.getElementById("refreshBtn").addEventListener("click", refreshAll);

document.getElementById("autoRefreshToggle").addEventListener("change", e => {
  autoRefresh = e.target.checked;
});

// Currency tabs
document.querySelectorAll(".chip").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentCurrency = btn.dataset.currency;
    refreshAll();
  });
});

// Mock Mode
document.getElementById("mockBtn").addEventListener("click", () => {
  state.mockMode = !state.mockMode;
  const btn = document.getElementById("mockBtn");
  btn.classList.toggle("active", state.mockMode);
  btn.textContent = state.mockMode ? "Mock Mode" : "Live Mode";
  refreshAll();
});

// Auto-refresh loop
setInterval(() => {
  if (autoRefresh) refreshAll();
}, 30000);

// ============================
// Init
// ============================
refreshAll();