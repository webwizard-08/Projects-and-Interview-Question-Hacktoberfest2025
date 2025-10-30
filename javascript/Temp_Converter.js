const readline = require("readline");

// Create input interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Asks for temperature value first
rl.question("Enter the temperature value: ", function (tempInput) {
  const temperature = parseFloat(tempInput);

  // Validate temperature input
  if (isNaN(temperature)) {
    console.log("Please enter a valid numeric temperature value!");
    rl.close();
    return;
  }

  // Asks for temperature unit
  rl.question("Enter the unit (celsius / fahrenheit / kelvin): ", function (unitInput) {
    const unit = unitInput.trim().toLowerCase();
    let result = "";

    // Conversion logic
    if (unit === "celsius") {
      const f = (temperature * 9 / 5) + 32;
      const k = temperature + 273.15;
      result = `${temperature}°C = ${f.toFixed(2)}°F and ${k.toFixed(2)}K`;
    } 
    else if (unit === "fahrenheit") {
      const c = (temperature - 32) * 5 / 9;
      const k = c + 273.15;
      result = `${temperature}°F = ${c.toFixed(2)}°C and ${k.toFixed(2)}K`;
    } 
    else if (unit === "kelvin") {
      const c = temperature - 273.15;
      const f = (c * 9 / 5) + 32;
      result = `${temperature}K = ${c.toFixed(2)}°C and ${f.toFixed(2)}°F`;
    } 
    else {
      console.log("Invalid unit! Please enter 'celsius', 'fahrenheit', or 'kelvin'.");
      rl.close();
      return;
    }

    // Display result
    console.log(result);
    rl.close();
  });
});