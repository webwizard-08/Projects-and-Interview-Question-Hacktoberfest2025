const dino = document.getElementById("dino");
const obstacle = document.getElementById("obstacle");
const scoreElement = document.getElementById("score");
const highScoreElement = document.getElementById("highscore");
const retryBtn = document.getElementById("retryBtn");
const startBtn = document.getElementById("startBtn");

let score = 0;
let highScore = localStorage.getItem("highscore") || 0;
highScoreElement.textContent = highScore;

let gameRunning = false;
let scoreInterval;
let collisionCheckInterval;

document.addEventListener("keydown", function (event) {
  if ((event.code === "Space" || event.code === "ArrowUp") && gameRunning && !dino.classList.contains("jump")) {
    dino.classList.add("jump");
    setTimeout(() => {
      dino.classList.remove("jump");
    }, 400);
  }
});

function startGame() {
  gameRunning = true;
  score = 0;
  scoreElement.textContent = score;
  retryBtn.style.display = "none";
  startBtn.style.display = "none";

  // Show obstacle
  obstacle.style.display = "block";
  obstacle.style.left = "100%";
  obstacle.style.animation = "moveObstacle 2s infinite linear";

  // Score logic
  scoreInterval = setInterval(() => {
    score++;
    scoreElement.textContent = score;
  }, 100);

  // Collision detection
  collisionCheckInterval = setInterval(() => {
    const dinoRect = dino.getBoundingClientRect();
    const obsRect = obstacle.getBoundingClientRect();
  
    // Tighter hitboxes
    const dinoBox = {
      left: dinoRect.left + 10,
      right: dinoRect.right - 10,
      top: dinoRect.top + 5,
      bottom: dinoRect.bottom - 5
    };
  
    const obsBox = {
      left: obsRect.left + 2,
      right: obsRect.right - 2,
      top: obsRect.top + 2,
      bottom: obsRect.bottom - 2
    };
  
    const isColliding = !(
      dinoBox.bottom < obsBox.top ||
      dinoBox.top > obsBox.bottom ||
      dinoBox.right < obsBox.left ||
      dinoBox.left > obsBox.right
    );
  
    if (isColliding) {
      gameOver();
    }
  }, 10);
  
}

// Random obstacle image on each loop
obstacle.addEventListener("animationiteration", () => {
  const rand = Math.random();
  if (rand < 0.5) {
    obstacle.src = "images/obstacle1.png";
    obstacle.style.width = "40px";
  } else {
    obstacle.src = "images/obstacle2.png";
    obstacle.style.width = "50px";
  }
});

function gameOver() {
  gameRunning = false;
  clearInterval(scoreInterval);
  clearInterval(collisionCheckInterval);
  obstacle.style.animation = "none";
  obstacle.style.left = window.getComputedStyle(obstacle).getPropertyValue("left");

  // Update High Score
  if (score > highScore) {
    highScore = score;
    localStorage.setItem("highscore", highScore);
    highScoreElement.textContent = highScore;
  }

  retryBtn.style.display = "block";
  alert("💥 Game Over! Try again.");
}

// Start game
startBtn.addEventListener("click", startGame);

// Retry game
retryBtn.addEventListener("click", () => location.reload());

//increase the speed
if (score % 5 === 0 && speed > 1000) {
    speed -= 100; 
  }

// day/night mode 
let isNight = false;

function toggleDayNight() {
  isNight = !isNight;
  document.body.className = isNight ? 'night' : 'day';
}

// Toggle every 20 seconds
setInterval(toggleDayNight, 20000);

