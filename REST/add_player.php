<?php

error_reporting(E_ALL);
ini_set('display_errors', 1);


// Establish database connection using PDO
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbName", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Database connection failed: " . $e->getMessage());
}

// Retrieve information from the request body
$requestBody = file_get_contents('php://input');

// Check if the request body is valid JSON
$isJson = json_decode($requestBody, true);
if (json_last_error() !== JSON_ERROR_NONE) {
    die("Invalid JSON data: " . json_last_error_msg());
}

// Extract player information from the request data
$team_id = $isJson['team_id'];
$surname = $isJson['surname'];
$given_names = $isJson['given_names'];
$nationality = $isJson['nationality'];
$date_of_birth = $isJson['date_of_birth'];

// Insert the new player into the database
$sql = "INSERT INTO Players (team_id, surname, given_names, nationality, date_of_birth) VALUES (?, ?, ?, ?, ?)";
$stmt = $pdo->prepare($sql);
$stmt->execute([$team_id, $surname, $given_names, $nationality, $date_of_birth]);

// Check if the player was inserted successfully
if ($stmt->rowCount() > 0) {
    echo "Player added successfully";
} else {
    echo "Error adding player";
}

?>

