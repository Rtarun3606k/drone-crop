const req = async () => {
  const result = await fetch(
    "https://city.imd.gov.in/citywx/responsive/api/fetchCity_static.php?ID=88820",
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    }
  );

  // You should also handle the response
  const data = await result.json();
  console.log(data);
  return data;
};

// Call the function to execute the request
req()
  .then((data) => {
    // Handle the data returned from the request
    console.log("Data received:", data);
  })
  .catch((error) => {
    // Handle any errors that occurred during the request
    console.error("Error fetching data:", error);
  });
