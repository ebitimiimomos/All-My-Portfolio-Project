<?php


// Establish database connection using PDO
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbName", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Database connection failed: " . $e->getMessage());
}

// Retrieve the team_id from the query string parameters
$team_id = isset($_GET['team_id']) ? $_GET['team_id'] : '';

// Prepare the SQL query to retrieve players of the specific team
$query = "SELECT * FROM Players WHERE team_id = :team_id";
$stmt = $pdo->prepare($query);
$stmt->bindParam(':team_id', $team_id);
$stmt->execute();

// Create an array to store player information
$players = array();

while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
    $player = $row;
    $players[] = $player;
}

// Convert the array of players to JSON format
$jsonResponse = json_encode($players);

// Set the Content-Type header to indicate JSON response
header('Content-Type: application/json');

// Output the JSON response
echo $jsonResponse;
?>
