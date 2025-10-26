// FunFactGenerator.js
// A random fun fact generator for developers with better structure and reusability

const funFacts = [
  "💡 JavaScript was created in just 10 days!",
  "🐞 The first computer bug was an actual moth found in a computer!",
  "🐧 Git was created by Linus Torvalds — the same person who made Linux.",
  "🐍 The name 'Python' came from a comedy group, not a snake!",
  "💻 VS Code is written in TypeScript — a superset of JavaScript.",
  "🧠 The first programmer in history was Ada Lovelace — in the 1800s!",
  "⚙️ The term ‘bug’ has been used in engineering since the 19th century!",
  "🌐 Tim Berners-Lee invented the World Wide Web in 1989.",
  "🧮 The first 1GB hard drive was announced in 1980 — it weighed 550 pounds!",
  "📱 The first iPhone had only 128MB of RAM."
];

// Function to return a random fun fact
function getRandomFact() {
  const randomIndex = Math.floor(Math.random() * funFacts.length);
  return funFacts[randomIndex];
}

// Function to display the fact nicely formatted
function displayFunFact() {
  console.log("✨ Developer Fun Fact of the Day ✨");
  console.log("-----------------------------------");
  console.log(getRandomFact());
  console.log("-----------------------------------");
}

// Run the generator if executed directly
if (require.main === module) {
  displayFunFact();
}

// Export for use in other modules
module.exports = { getRandomFact, displayFunFact };

