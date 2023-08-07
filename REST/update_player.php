<?php

// Establish database connection using PDO
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbName", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Database connection failed: " . $e->getMessage());
}

if ($_SERVER['REQUEST_METHOD'] === 'PUT') {
    // Retrieve the JSON payload from the request body
    $requestBody = file_get_contents('php://input');
    $jsonData = json_decode($requestBody, true);

    // Check if the JSON payload is valid
    if (json_last_error() !== JSON_ERROR_NONE) {
        die("Invalid JSON data: " . json_last_error_msg());
    }

    // Get the player data from the JSON payload
    $team_id = $jsonData['team_id'];
    $player_id = $jsonData['player_id'];
    $surname = $jsonData['surname'];
    $given_names = $jsonData['given_names'];
    $nationality = $jsonData['nationality'];
    $date_of_birth = $jsonData['date_of_birth'];

    // Check if the player data is valid
    if (empty($team_id) || !is_numeric($team_id)) {
        die("Please provide a valid team ID.");
    }
    if (empty($player_id) || !is_numeric($player_id)) {
        die("Please provide a valid player ID.");
    }
    if (empty($surname)) {
        die("Please provide the player's updated surname.");
    }
    if (empty($given_names)) {
        die("Please provide the player's updated given names.");
    }
    if (empty($nationality)) {
        die("Please provide the player's updated nationality.");
    }
    if (empty($date_of_birth)) {
        die("Please provide the player's updated date of birth.");
    }

    // Update the player in the database
    $sql = "UPDATE Players SET surname = ?, given_names = ?, nationality = ?, date_of_birth = ?, team_id = ? WHERE player_id = ?";
    $stmt = $pdo->prepare($sql);
    $stmt->execute([$surname, $given_names, $nationality, $date_of_birth, $team_id, $player_id]);

    // Check if the player was updated successfully
    if ($stmt->rowCount() > 0) {
        echo "Player updated successfully";
    } else {
        echo "Error updating player";
    }
}
?>