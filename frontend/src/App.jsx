import { useMemo, useRef, useState } from "react";
import { predictRisk } from "./api";

const numericFields = [
  { name: "hour_from_admission", label: "Hour from admission", example: "Example: 5", help: "Hours since patient was admitted." },
  { name: "heart_rate", label: "Heart rate (bpm)", example: "Example: 112", help: "Beats per minute." },
  { name: "respiratory_rate", label: "Respiratory rate (breaths/min)", example: "Example: 24", help: "Breaths per minute." },
  { name: "spo2_pct", label: "SpO2 (%)", example: "Example: 91", help: "Peripheral oxygen saturation percentage." },
  { name: "temperature_c", label: "Temperature (°C)", example: "Example: 38.5", help: "Body temperature in Celsius." },
  { name: "systolic_bp", label: "Systolic BP (mmHg)", example: "Example: 95", help: "Upper blood pressure value." },
  { name: "diastolic_bp", label: "Diastolic BP (mmHg)", example: "Example: 62", help: "Lower blood pressure value." },
  { name: "oxygen_flow", label: "Oxygen flow (L/min)", example: "Example: 4", help: "Supplemental oxygen flow rate." },
  { name: "mobility_score", label: "Mobility score (0-5)", example: "Example: 2", help: "Clinical mobility scale value." },
  { name: "wbc_count", label: "WBC count", example: "Example: 13.2", help: "White blood cell count." },
  { name: "lactate", label: "Lactate", example: "Example: 2.8", help: "Serum lactate." },
  { name: "creatinine", label: "Creatinine", example: "Example: 1.6", help: "Serum creatinine." },
  { name: "crp_level", label: "CRP level", example: "Example: 56", help: "C-reactive protein level." },
  { name: "hemoglobin", label: "Hemoglobin", example: "Example: 10.8", help: "Hemoglobin concentration." },
  { name: "age", label: "Age", example: "Example: 71", help: "Patient age in years." },
  { name: "comorbidity_index", label: "Comorbidity index", example: "Example: 3", help: "Comorbidity burden score." }
];

const categoricalFields = [
  {
    name: "gender",
    label: "Gender",
    help: "Patient biological sex used in training schema.",
    options: ["Male", "Female"]
  },
  {
    name: "admission_type",
    label: "Admission type",
    help: "Clinical admission category.",
    options: ["Emergency", "Elective", "Urgent"]
  },
  {
    name: "oxygen_device",
    label: "Oxygen device",
    help: "Current oxygen support mode.",
    options: ["None", "Nasal Cannula", "Mask", "Ventilator"]
  }
];

const initialState = {
  hour_from_admission: "",
  heart_rate: "",
  respiratory_rate: "",
  spo2_pct: "",
  temperature_c: "",
  systolic_bp: "",
  diastolic_bp: "",
  oxygen_flow: "",
  mobility_score: "",
  wbc_count: "",
  lactate: "",
  creatinine: "",
  crp_level: "",
  hemoglobin: "",
  age: "",
  comorbidity_index: "",
  gender: "Male",
  admission_type: "Emergency",
  oxygen_device: "None",
  nurse_alert: false
};

const featureLabels = {
  heart_rate: "heart rate",
  respiratory_rate: "respiratory rate",
  spo2_pct: "oxygen saturation (SpO2)",
  temperature_c: "temperature",
  systolic_bp: "systolic blood pressure",
  diastolic_bp: "diastolic blood pressure",
  oxygen_flow: "oxygen flow",
  mobility_score: "mobility score",
  wbc_count: "WBC count",
  lactate: "lactate",
  creatinine: "creatinine",
  crp_level: "CRP level",
  hemoglobin: "hemoglobin",
  age: "age",
  comorbidity_index: "comorbidity index",
  NEWS2: "NEWS2 score",
  sofa_proxy: "organ stress proxy",
  hour_from_admission: "time since admission"
};

const clinicalPrecautions = {
  High: [
    "Escalate immediately to physician or rapid response team.",
    "Repeat full vital signs and focused clinical assessment now.",
    "Increase monitoring frequency (for example every 15-30 minutes).",
    "Prepare oxygen/airway support and emergency medications per protocol."
  ],
  Medium: [
    "Reassess vital signs and symptoms within 30-60 minutes.",
    "Review fluid status, oxygen support, and lab trends with the care team.",
    "Keep patient under closer observation and watch for worsening signs.",
    "Escalate early if lactate, SpO2, blood pressure, or respiratory status declines."
  ],
  Low: [
    "Continue routine monitoring and nursing rounds.",
    "Re-check inputs if clinical picture changes.",
    "Maintain current care plan and standard safety checks.",
    "Escalate if new warning signs appear despite current low risk score."
  ]
};

function toClinicalLabel(key) {
  if (featureLabels[key]) {
    return featureLabels[key];
  }
  return key.replace(/_/g, " ");
}

function App() {
  const [formData, setFormData] = useState(initialState);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const fieldOrder = useMemo(
    () => [
      ...numericFields.map((f) => f.name),
      ...categoricalFields.map((f) => f.name),
      "nurse_alert",
      "submit"
    ],
    []
  );

  const refs = useRef({});

  const onEnterNext = (name, event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    const idx = fieldOrder.indexOf(name);
    const nextName = fieldOrder[idx + 1];
    const nextEl = refs.current[nextName];
    if (nextEl && typeof nextEl.focus === "function") {
      nextEl.focus();
    }
  };

  const onChange = (name, value) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        ...formData,
        nurse_alert: Boolean(formData.nurse_alert)
      };

      for (const field of numericFields) {
        payload[field.name] = Number(payload[field.name]);
      }

      const data = await predictRisk(payload);
      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to get prediction.");
    } finally {
      setLoading(false);
    }
  };

  const riskClass = (result?.risk_level || "").toLowerCase();

  return (
    <div className="page-shell">
      <main className="card">
        <header className="header-block">
          <p className="eyebrow">AI Clinical Safety</p>
          <h1>Early Warning for Physiological Deterioration</h1>
          <p className="subtitle">
            Enter bedside observations to get a clear risk summary, clinical explanations, and immediate precautions.
          </p>
        </header>

        <form onSubmit={onSubmit} className="form-grid">
          {numericFields.map((field) => (
            <label className="input-group" key={field.name}>
              <span className="label-row">
                <span>{field.label}</span>
                <span className="help" title={field.help}>?</span>
              </span>
              <input
                ref={(el) => {
                  refs.current[field.name] = el;
                }}
                type="number"
                step="any"
                value={formData[field.name]}
                onChange={(e) => onChange(field.name, e.target.value)}
                onKeyDown={(e) => onEnterNext(field.name, e)}
                required
              />
              <small>{field.example} - {field.help}</small>
            </label>
          ))}

          {categoricalFields.map((field) => (
            <label className="input-group" key={field.name}>
              <span className="label-row">
                <span>{field.label}</span>
                <span className="help" title={field.help}>?</span>
              </span>
              <select
                ref={(el) => {
                  refs.current[field.name] = el;
                }}
                value={formData[field.name]}
                onChange={(e) => onChange(field.name, e.target.value)}
                onKeyDown={(e) => onEnterNext(field.name, e)}
              >
                {field.options.map((opt) => (
                  <option value={opt} key={opt}>
                    {opt}
                  </option>
                ))}
              </select>
              <small>{field.help}</small>
            </label>
          ))}

          <label className="input-group checkbox-row">
            <span className="label-row">
              <span>Nurse alert</span>
              <span className="help" title="Set Yes if nursing concern is currently raised.">?</span>
            </span>
            <div className="inline-toggle">
              <input
                ref={(el) => {
                  refs.current.nurse_alert = el;
                }}
                type="checkbox"
                checked={formData.nurse_alert}
                onChange={(e) => onChange("nurse_alert", e.target.checked)}
                onKeyDown={(e) => onEnterNext("nurse_alert", e)}
              />
              <span>{formData.nurse_alert ? "Yes" : "No"}</span>
            </div>
          </label>

          <button
            ref={(el) => {
              refs.current.submit = el;
            }}
            type="submit"
            className="submit-btn"
            disabled={loading}
          >
            {loading ? "Calculating..." : "Predict Risk"}
          </button>
        </form>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <section className="result-panel">
            <div className={`risk-badge ${riskClass}`}>
              <h2>{result.risk_level} Risk</h2>
              <p>Risk Score: {result.risk_score_pct}%</p>
            </div>

            <div className="component-grid">
              <article className="metric-card">
                <h3>Clinical Interpretation</h3>
                <p>{result.explanations.shap.summary}</p>
              </article>

              <article className="metric-card">
                <h3>Why This Risk Is Elevated</h3>
                <p>{result.explanations.lime.summary}</p>
              </article>

              <article className="metric-card">
                <h3>Recommended Precautions</h3>
                <ul>
                  {(clinicalPrecautions[result.risk_level] || clinicalPrecautions.Low).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
            </div>

            <div className="explain-grid">
              <article className="explain-card">
                <h3>Top Factors Affecting This Patient</h3>
                <ul>
                  {result.explanations.shap.local_top_contributors.map((item) => (
                    <li key={item.feature}>
                      <strong>{toClinicalLabel(item.feature)}</strong>: {item.impact === "increase" ? "raising" : "lowering"} current risk
                    </li>
                  ))}
                </ul>
              </article>

              <article className="explain-card">
                <h3>Most Important Clinical Signals (Overall)</h3>
                <ul>
                  {result.explanations.shap.global_top_features.map((item) => (
                    <li key={item.feature}>
                      <strong>{toClinicalLabel(item.feature)}</strong>
                    </li>
                  ))}
                </ul>
              </article>

              <article className="explain-card">
                <h3>Patient-Specific Reasoning</h3>
                <ul>
                  {result.explanations.lime.local_rules.map((rule) => (
                    <li key={rule.rule}>
                      {rule.rule}
                    </li>
                  ))}
                </ul>
              </article>
            </div>

            <article className="metric-card">
              <h3>Nursing Note</h3>
              <p>
                This tool supports clinical judgment and escalation decisions. Always combine this output with bedside assessment and hospital protocol.
              </p>
            </article>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;