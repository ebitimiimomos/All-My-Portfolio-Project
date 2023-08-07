//declares these variables and initializes it with an empty string.
var requestMethod = "";
var resourceURL = "";
var requestBody = "";

//defines the function to toggle the resource list
function toggleResourceList() {

//retrieves element from the HTML document assigns it to the variable
const resourceList = document.getElementById("resourceList"); 
//toggles the presence of the CSS class "toggle-content" 
resourceList.classList.toggle("toggle-content");
}
    
//defines the function named RequestMethod 
function RequestMethod(method) {
  requestMethod = method; //assigns the value of the parameter to the variable
  updateDatalistOptions();
  displayRequestBody();
}

// Function to update the options in the datalist based on the selected request method
function updateDatalistOptions() {
  // Retrieve elements from the HTML document and assign them to variables
  const datalist = document.getElementById("resources");
  const resourceInput = document.getElementById("resource");
  resourceInput.value = "";

  var datalistOptions = [];

  // Populate the datalist options with all available resources
  datalistOptions = [

  ];

// Create and append option elements to the datalist based on the datalist options
  for (var k = 0; k < datalistOptions.length; k++) {
    const res = datalistOptions[k];
    const option = document.createElement('option');
    option.value = res;
    datalist.appendChild(option);
  }
}

// Function to set the resource URL
function setResourceURL(input) {
  resourceURL = input;
}

// Function to set the request body
function setRequestBody(input) {
  requestBody = input;
}



// Function to retrieve the response from the server
function retrieveResponse() {
// Create a new XMLHttpRequest object
  const request = new XMLHttpRequest();
  // Define the callback function to handle the response
  request.onreadystatechange = function () {
  // Check if the request is complete and the response is ready
    if (this.readyState === 4) {      
      
      // Update the status code display
    document.getElementById("statusCode").innerHTML =
        "HTTP Status Code: " + this.status + " " + this.statusText + "<br><br>";
      // Check if the response status is 200 (OK)
      if (this.status === 200) {
        try {
        // Parse the response as JSON
          const responseJSON = JSON.parse(this.responseText);
          
         // Format the JSON response for display
          const formattedResponse = JSON.stringify(responseJSON, null, 2)
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\n/g, "<br>")
            .replace(/\s/g, "&nbsp;");

          // Show the response body and display the formatted response
          document.getElementById("responseBody").innerHTML = "HTTP Response Body:" + formattedResponse;
          document.getElementById("responseBody").style.display = "block";

          
          // Print "Request successful!"
        console.log("Request successful!, Clear Page to make another request");
        
        } catch (error) {
          
          // Show the response body and display the raw response
          document.getElementById("responseBody").innerHTML = this.responseText;
          document.getElementById("responseBody").style.display = "block";
        }
      } else {
        // Hide the response body and display the raw response
        document.getElementById("responseBody").innerHTML = "Error: " + this.responseText;
        document.getElementById("responseBody").style.display = "none";
      }
    }
  };

  // // Open the request with the specified method and URL
    const requestMethod = document.getElementById("method").value;
  const resourceURL = document.getElementById("resource").value;
  request.open(requestMethod, resourceURL, true);
  
  // Send the request to the server
  if (requestMethod === "GET") {
    request.send();
  } else {
  // Set the request header for JSON content type
    request.setRequestHeader("Content-Type", "application/json");
    // Send the request body with the request
    request.send(requestBody);
  }
}



// Function to clear the response and reload the page
function clearResponse() {
  // Reload the current page
  location.reload();
}