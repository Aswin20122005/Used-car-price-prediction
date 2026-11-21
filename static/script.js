const form = document.getElementById("priceForm");
const result = document.querySelector(".prediction-result");

const FIELDS = [
  "make", "year", "kilometers_driven", "fuel_type", "transmission", "owner_type",
  "mileage_kmpl", "engine_cc", "power_bhp", "seats",
];

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  // Collect the form data (blank optional fields are sent as "" and filled in by the API)
  const carData = {};
  FIELDS.forEach((id) => (carData[id] = document.getElementById(id).value));

  try {
    // Send the POST request to the backend
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(carData),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to fetch prediction");
    result.innerText = `Predicted Price: ₹${data.predicted_price} Lakh`;
  } catch (error) {
    console.error("Error:", error);
    result.innerText = `Error: ${error.message}`;
  }
});
