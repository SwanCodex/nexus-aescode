export async function predictRisk(payload) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const msg = await response.text();
    throw new Error(msg || "Prediction failed");
  }

  return response.json();
}