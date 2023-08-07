<!DOCTYPE html>
<html lang="en-GB">
  <head>
    <title>IT training sessions</title>
    <link rel="stylesheet" type="text/css" href="training.css">
    
  </head>
  <body>
  <header>
  <div class="logo">
    <img src="https://www.liverpool.ac.uk/logo-size-test/full-colour.svg" alt="Logo">
    <div class="header">
      <h1>University IT Training </h1>
    </div>
    </div>
    </header>
    </div>
    


<?php



// Constructs the PDO object with connection details
$dsn = "mysql:host=$db_hostname;dbname=$db_database;charset=$db_charset";
$opt = array(
    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    PDO::ATTR_EMULATE_PREPARES => false
);
 
try {
    $pdo = new PDO($dsn, $db_username, $db_password, $opt);
    

      
// Checks if the topic is not set in the POST request
if (!isset($_POST['topic'])) {
// Retrieve the distinct topics with available capacity from the database and order them alphabetically
    $stmt = $pdo->query("SELECT DISTINCT topic FROM training_sessions WHERE Capacity > 0 ORDER BY topic ASC");
    $topic = $stmt->fetchAll();


// Displays a form for the user to select a topic
    echo '<form name="form" method="post">
        <h2>Let\'s get you booked in!</h2> 
        <label for="topic">Select a topic:</label>
        <select name="topic" id="topic" required>
            <option value="">Select a topic</option>';

    // Loops through the retrieved topics and display each one as an option in the dropdown menu
    foreach ($topic as $topic) {
    // Retrieve the capacity of the topic from the database
        $capacity = $pdo->query("SELECT Capacity FROM training_sessions WHERE topic = '$topic[topic]'")->fetchColumn();
        // Only display the topic in the dropdown if there is available capacity for it
        if ($capacity > 0) {
        echo '<option value="' . $topic['topic'] . '">' . $topic['topic'] . '</option>';
    }
  }
  // Close the dropdown menu and display the form submit button
    echo '</select>
          </div>
          <input type="submit" value="Next">
          </form>';
  
  
 } else { // If the topic is set in the POST request
    // Get the selected topic from the POST request
    $topic = $_POST['topic'];
    // Retrieve the capacity of the selected topic from the database and cast it to an integer
    $capacity = (int) $row['capacity'];
    
    // Query the database for all training sessions that match the selected topic
    $stmt = $pdo->query("SELECT id, day, time FROM training_sessions WHERE topic = '$topic'");
    $start_times = $stmt->fetchAll();

    // Display a form to allow the user to select a training session to attend
    echo '<form name="form" method="post">
          <h2>Almost There!</h2>
          <label for="topic">You Have Selected:</label>
          <input type="text" name="topic" value="' . $topic . '" readonly>
          <br>
          <br>
          <label for="start_time">Select a session to attend:</label>
          <select name="start_time" id="start_time" required>
              <option value="">pick a session(subject to availability)</option>';

    // Loop through each training session and add an option to the dropdown menu for each one
    foreach ($start_times as $start_time) {
        // Query the database to check if the training session has any available capacity
        $capacity = $pdo->query("SELECT Capacity FROM training_sessions WHERE id = '$start_time[id]'")->fetchColumn();
        
        // If the training session has available capacity, add an option to the dropdown menu for the user to select it
        if ($capacity > 0) {
            echo '<option value="' . $start_time['day'] . ' ' . $start_time['time'] . '">' 
        . $start_time['day'] . ' - ' . $start_time['time'] . '</option>';
       
       // If the training session is full, add an option to the dropdown menu to inform the user that it is full
        } else {
            echo '<option value="' . $start_time['day'] . ' ' . $start_time['time'] . '">' 
        . $start_time['day'] . ' - ' . $start_time['time'] . '</option>';
            echo '<p>All sessions for this topic are full.</p>';
        }
    }


        // input fields for the user to enter their name and email, and buttons to go back and submit the form
    echo '</select>
          <br>
          <br>
          <label for="name">Your Name:</label>
          <input type="text" name="name" placeholder="Enter your name here"id="name" required>
          <br>
          <br>
          <label for="email">Your Email:</label>
          <input type="text" name="email" placeholder="Enter your email here" id="email" required>
          <br>
          <br>
          <a href="training.php" class="button">Go Back</a>
          <input type="submit"name="submit" value="Submit">
          </form>';
}

    
    
    if (isset($_POST['submit'])) {
      // Retrieve the submitted values
      $name = $_POST['name'];
      $email = $_POST['email'];
      $start_time = $_POST['start_time'];
  
  
      // Check if the name is valid using regular expressions
     if (!preg_match('/^[a-zA-Z\'\-\s]+$/', $name)) {
          $error_message = "The booking attempt was unsuccessful due to an invalid name entry.";
      }
      
      // Check if the name contains any SQL injection attempt
      if (preg_match('/(\'\')|(--)/', $name)) {
          $error_message = "The booking attempt was unsuccessful due to an invalid name entry.";
      }
 
       // Check if the name starts with letters only
      if (!preg_match('/^[a-zA-Z\']/', $name)) {
          $error_message = "The booking attempt was unsuccessful due to an invalid name entry.";
      }
      
      // Check if the name ends with letters or digits only
      if (preg_match('/[\s\-]$/', $name)) {
          $error_message = "The booking attempt was unsuccessful due to an invalid name entry.";
      }
      
      // Check if the email is valid using filter_var function
      if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
          $error_message = "The booking attempt was unsuccessful due to an invalid Email entry.";
      }
      

      // Display the error message if any
      if (!empty($error_message)) {
  echo '<p class="error">' . $error_message . '</p>';
} else {

      // Insert the data into the database using prepared statements to prevent SQL injection
$stmt = $pdo->prepare("INSERT INTO booking_request (name, email, topic, time) VALUES (?,?, ?, ?)");
$stmt->execute(array($name, $email, $topic, $start_time));
       

  // Update the number of available places in the capacity
    $stmt = $pdo->prepare("UPDATE training_sessions SET capacity = capacity - 1 WHERE topic = :topic AND time = :start_time");
    $stmt->bindParam(':topic', $topic);
    $stmt->bindParam(':start_time', $start_time);
    $stmt->execute();
  
  
    
  // Display a confirmation details to student
  echo '<div class="confirmation-box">';
  echo '<h2>Your booking has been confirmed!</h2>';
  echo '<p>Name: ' . $name . '</p>';
  echo '<p>Email: ' . $email . '</p>';
  echo "Session Topic: " . $topic;
  echo '<p>Session Time: ' . $start_time . '</p>';
  echo '<a href="training.php" class="button">Make Another Booking</a>';
  echo '</div>';
  
}

      }
  }
 
 // End of the code block and error handling if any
catch (PDOException $e) {
    echo $e->getMessage();
}

?>


  
  </body>
</html>