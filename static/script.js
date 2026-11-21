const form = document.getElementById("priceForm");
const result = document.querySelector(".prediction-result");

// Fill the Make dropdown with the brands the model was trained on.
async function loadOptions() {
  try {
    const response = await fetch("/options");
    if (!response.ok) return;
    const options = await response.json();
    const make = document.getElementById("make");
    options.Brand.forEach((brand) => make.add(new Option(brand, brand)));
  } catch (error) {
    console.error("Could not load options:", error);
  }
}
loadOptions();

const FIELDS = [
  "make", "year", "kilometers_driven", "fuel_type", "transmission", "owner_type",
  "mileage_kmpl", "engine_cc", "power_bhp", "seats",
];

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  // Collect the form data (blank optional fields are sent as "" and filled in by the API)
  const carData = {};
  FIELDS.forEach((id) => (carData[id] = document.getElementById(id).value));

  const button = form.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "Predicting…";
  result.innerText = "…";

  try {
    // Send the POST request to the backend
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(carData),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to fetch prediction");
    const rupees = Math.round(data.predicted_price * 100000).toLocaleString("en-IN");
    result.innerText = `Predicted Price: ₹${data.predicted_price} Lakh (≈ ₹${rupees})`;
    document.getElementById("result-section").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    console.error("Error:", error);
    result.innerText = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Get Prediction";
  }
});

document.getElementById("year").textContent = new Date().getFullYear();
