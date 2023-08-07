<?php


// Establish database connection using PDO
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbName", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Database connection failed: " . $e->getMessage());
}

if ($_SERVER['REQUEST_METHOD'] === 'DELETE') {
    // Retrieve the player ID from the request body
    $requestBody = file_get_contents('php://input');
    $isJson = json_decode($requestBody, true);

    // Check if the request body is valid JSON
    if (json_last_error() !== JSON_ERROR_NONE) {
        die("Invalid JSON data: " . json_last_error_msg());
    }

    // Extract the player ID from the request data
    $playerId = $isJson['player_id'];

    // Check if the player ID is valid
    if (!is_numeric($playerId)) {
        echo "Please enter a valid player ID.";
        return;
    }

    // Check if the player ID exists in the database
    $selectStmt = $pdo->prepare("SELECT * FROM Players WHERE player_id = :id");
    $selectStmt->bindParam(':id', $playerId, PDO::PARAM_INT);
    $selectStmt->execute();

    // Check if the player exists
    if ($selectStmt->rowCount() > 0) {
        // Prepare and execute the delete query
        $deleteStmt = $pdo->prepare("DELETE FROM Players WHERE player_id = :id");
        $deleteStmt->bindParam(':id', $playerId, PDO::PARAM_INT);

        // Check if the delete query execution is successful
        if ($deleteStmt->execute()) {
            if ($deleteStmt->rowCount() > 0) {
                echo "Player with ID $playerId has been deleted successfully.";
            } else {
                echo "Failed to delete player with ID $playerId.";
            }
        } else {
            print_r($deleteStmt->errorInfo());
            echo "Error executing delete query.";
        }
    } else {
        echo "Player with ID $playerId does not exist.";
    }
}
?>
