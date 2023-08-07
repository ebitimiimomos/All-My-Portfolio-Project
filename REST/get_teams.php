<?php


// Establish database connection using PDO
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbName", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Database connection failed: " . $e->getMessage());
}

// Retrieve teams from the database
$query = "SELECT * FROM Teams";
$stmt = $pdo->query($query);

// Create an array to store team information
$teams = array();

while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
    $team = $row;
    $teams[] = $team;
}

// Convert the array of teams to JSON format
$jsonResponse = json_encode($teams);

// Set the Content-Type header to indicate JSON response
header('Content-Type: application/json');

// Output the JSON response
echo $jsonResponse;
?>
