// Set Up Stage
function setupGame() {
  // Parse the number of dice entered by the user
  numDice = parseInt(document.getElementById("numDiceInput").value);

  // If the entered value is not a natural number between 3 and 6 or it's a decimal value, show an error message
  if (isNaN(numDice) || !Number.isInteger(numDice) || numDice < 3 || numDice > 6 || document.getElementById("numDiceInput").value.includes(".")) {
    document.getElementById("errorMsg").textContent = "Please enter a natural number between 3 and 6!";
  } else { // If the entered value is a valid input, hide the setup section, show the play section, and set the round number to 1
    document.getElementById("setup").style.display = "none";
    document.getElementById("play").style.display = "block";
    document.getElementById("roundNum").textContent = 1;
  }
}

let balance = 0;
let numRounds = 0;
let totalPoints = 0;
const scores = [];

// Play Stage.
/*This function plays a round of the game by rolling the dice, calculating the points won, updating the balance and total points, and displaying the dice and points won.*/
function playRound() {
  // Roll the dice and calculate the points won
  const diceValues = [];
  for (let i = 0; i < numDice; i++) {
    const roll = Math.floor(Math.random() * 6) + 1;
    diceValues.push(roll);
  }
  const pointsWon = calculateScore(diceValues);
  // Update the balance, number of rounds, and total points
  balance += pointsWon;
  numRounds++;
  totalPoints += pointsWon;
  scores.push(pointsWon);
  // Update the round number, display the dice, and show the points won and balance
  document.getElementById("roundNum").textContent = numRounds;
  document.getElementById("dice").innerHTML = "";
  for (let i = 0; i < numDice; i++) {
    const die = document.createElement("img");
    die.classList.add("die");
	// Here the image of the die face images are linked to the numbers so that they display instead of an interger. 
	//This images are gotten from a copyright free website and below are the links:
	// https://commons.wikimedia.org/wiki/File:Dice-1-b.svg
	// https://commons.wikimedia.org/wiki/File:Dice-2-b.svg
	// https://commons.wikimedia.org/wiki/File:Dice-3-b.svg
	// https://commons.wikimedia.org/wiki/File:Dice-4-b.svg
	// https://commons.wikimedia.org/wiki/File:Dice-5-b.svg
	// https://commons.wikimedia.org/wiki/File:Dice-6-b.svg
	
    die.setAttribute("src", "dice-" + diceValues[i] + ".png");
    document.getElementById("dice").appendChild(die);
  }
  document.getElementById("pointsWon").textContent = "Points won: " + pointsWon;
  document.getElementById("balance").textContent = "Balance: " + balance;
}

function calculateScore(diceValues) {
  const sortedValues = diceValues.slice().sort();
  const sum = sortedValues.reduce((total, num) => total + num, 0);
  let sameValues = 0;
  let maxSameValues = 0;
  let isStraight = true;

  for (let i = 0; i < sortedValues.length - 1; i++) {
    if (sortedValues[i] === sortedValues[i + 1]) {
      sameValues++;
      maxSameValues = Math.max(maxSameValues, sameValues);
      isStraight = false;
    } else {
      sameValues = 0;
    }

    if (sortedValues[i] !== sortedValues[i + 1] - 1) {
      isStraight = false;
    }
  }

  let score = sum;

  if (isStraight) {
    score += 20;
  }

  if (maxSameValues === sortedValues.length - 1) {
    score += 60;
  } else if (maxSameValues === sortedValues.length - 2) {
    score += 40;
  }

  return score;
}


function endGame() {
  document.getElementById("play").style.display = "none";
  document.getElementById("end").style.display = "block";
  document.getElementById("numRounds").textContent = numRounds;
  document.getElementById("totalPoints").textContent = totalPoints;
  let avgPoints = totalPoints / numRounds;
  document.getElementById("avgPoints").innerHTML =  avgPoints.toFixed(1);
  drawGraph(scores);
}

function playAgain() {
  balance = 0;
  numRounds = 0;
  totalPoints = 0;
  document.getElementById("setup").style.display = "block";
  document.getElementById("play").style.display = "none";
  document.getElementById("end").style.display = "none";
  location.reload();
}

function drawGraph(scores) {
  let canvas = document.getElementById("graph");
  let context = canvas.getContext("2d");
  let width = 600; // canvas size 
  let height = 500; // canvas height
  let originX = 30; // origin and axis labels
  let originY =370;
  let maxScore = Math.max(...scores);
  let unitScore = (height -100) / maxScore;
  let unitWidth = width / scores.length;
  let barWidth = 0.4 * unitWidth; // width of each bar
  
  // clear the canvas
  context.clearRect(0, 0, canvas.width, canvas.height);
  
  // x and y axis with grid lines
  context.beginPath();
  context.strokeStyle = "rgb(0, 0, 0)";
  context.moveTo(originX, originY);
  context.lineTo(originX, 10);
  context.stroke();
  context.moveTo(originX, originY);
  context.lineTo(originX + width + 20, originY);
  context.stroke();
  for(let i=1; i<=maxScore; i++){
    context.beginPath();
    context.strokeStyle = "rgb(220, 220, 220)";
    context.moveTo(originX, originY - i*unitScore);
    context.lineTo(originX + width, originY - i*unitScore);
    context.stroke();
  }
  context.closePath();

  // axis labels
  context.fillStyle = "rgb(0, 0, 0)";
  context.font = "bold 10pt Arial";
  context.textAlign = "center";
  context.fillText("Round", originX + width/2, originY + 50);
  context.fillText("Score", originX + width/3, originY + -345);

  // plot data points as bars
  context.fillStyle = "rgb(0, 120, 200)";
  for (let i = 0; i < scores.length; i++) {
    let x = originX + i * unitWidth + (unitWidth - barWidth) / 2;
    let y = originY - scores[i] * unitScore;
    let barHeight = scores[i] * unitScore;
    context.fillRect(x, y, barWidth, barHeight);
    context.strokeRect(x, y, barWidth, barHeight);

    context.fillStyle = "rgb(0, 120, 200)";
    context.fillText(i + 1, x + barWidth / 2, originY + 25);
  }
  
  // add legend
  context.beginPath();
  context.moveTo(originX + width - 40, 55);
  context.lineTo(originX + width - 70, 55);
  context.stroke();
  context.closePath();
}
