<?php


// Establish database connection using PDO
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbName", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Database connection failed: " . $e->getMessage());
}

// Retrieve the playerID from the query string parameters
$player_id = isset($_GET['player_id']) ? $_GET['player_id'] : '';

// Prepare the SQL query to retrieve information of the specific player
$query = "SELECT * FROM Players WHERE player_id = :player_id";
$stmt = $pdo->prepare($query);
$stmt->bindParam(':player_id', $player_id);
$stmt->execute();

// Fetch the player information
$player = $stmt->fetch(PDO::FETCH_ASSOC);

// Set the Content-Type header to indicate JSON response
header('Content-Type: application/json');

// Output the JSON response
echo json_encode($player);
?>
