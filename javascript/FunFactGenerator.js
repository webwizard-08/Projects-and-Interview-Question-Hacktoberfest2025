// FunFactGenerator.js
// A simple random fun fact generator for developers

const funFacts = [
  "JavaScript was created in just 10 days!",
  "The first computer bug was an actual moth.",
  "Git was created by Linus Torvalds — the same person who made Linux.",
  "The name 'Python' came from a comedy group, not a snake!",
  "VS Code is written in TypeScript — a superset of JavaScript."
];

function getRandomFact() {
  const randomIndex = Math.floor(Math.random() * funFacts.length);
  return funFacts[randomIndex];
}

console.log("💡 Developer Fun Fact:");
console.log(getRandomFact());
